import torch
from tqdm import tqdm
from info_nce import InfoNCE
from pytorch_metric_learning import losses, distances


class PreTrainer():
    def __init__(self, epochs, device, transform):
        self.epochs = epochs
        self.device = device
        self.loss_fnc = InfoNCE(negative_mode='paired')
        self.transform = transform

    def fit(self, model, data_loader, optimizer, lr_scheduler):
        best_loss = 10
        for epoch in range(1, self.epochs + 1):
            print('Epoch {}'.format(epoch))
            model.train()
            running_loss = 0.0
            with tqdm(data_loader, unit="batch") as data_loader:
                for batch in data_loader:
                    batch_size = batch[0].size()[0]
                    batch = torch.cat(batch)
                    batch = [list(b) for b in list(batch)]
                    batch = self.transform(batch, return_tensors="pt")
                    batch.to(self.device)

                    sequence_output = model(**batch)[0]
                    logits = sequence_output[:, 0]

                    query = logits[:batch_size]
                    positive_key = logits[batch_size:int(batch_size*2)]
                    negative_key = logits[int(batch_size*2):int(batch_size * 3)]
                    intra_negative_key = logits[int(batch_size * 3):]

                    negative_keys = negative_key.unsqueeze(1)

                    negative_keys_smile = []
                    if batch_size == 1:
                        negative_keys_smile = intra_negative_key.unsqueeeze(1)
                    else:
                        for i in range(batch_size):
                            i_inter = i+1 if i+1<batch_size else 0
                            negative_keys_smile.append(torch.stack([intra_negative_key[i], positive_key[i_inter]], 0))
                        negative_keys_smile = torch.stack(negative_keys_smile, 0)

                    kin_loss = self.loss_fnc(query, positive_key, negative_keys)
                    smile_loss = self.loss_fnc(query, positive_key, negative_keys_smile)
                    loss = 0.6*kin_loss + 0.4*smile_loss

                    optimizer.zero_grad()
                    loss.backward()
                    lr_scheduler.step()
                    optimizer.step()

                    running_loss += loss.item()
                    data_loader.set_postfix(loss=loss.item(), loss_kin=kin_loss.item(), loss_smile=smile_loss.item())
            avg_loss = running_loss / len(data_loader)
            print('Loss {}'.format(avg_loss))

            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_pretrained('./tmp/.')

    def validate(self, model, data_loader):
        model.to(self.device)
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            with tqdm(data_loader, unit="batch") as data_loader:
                for batch in data_loader:
                    batch_size = batch[0].size()[0]
                    batch = torch.cat(batch)
                    batch = [list(b) for b in list(batch)]
                    batch = self.transform(batch, return_tensors="pt")
                    batch.to(self.device)

                    sequence_output = model(**batch)[0]
                    logits = sequence_output[:, 0]

                    query = logits[:batch_size]
                    positive_key = logits[batch_size:int(batch_size*2)]
                    negative_key = logits[int(batch_size*2):int(batch_size * 3)].unsqueeze(1)

                    loss = self.loss_fnc(query, positive_key, negative_key)
                    running_loss += loss.item()
                    data_loader.set_postfix(loss=loss.item())
        avg_loss = running_loss / len(data_loader)
        print('val Loss {}'.format(avg_loss))

    def predict(self, model, data_loader):
        model.to(self.device)
        model.eval()

        predictions = []
        with torch.no_grad():
            with tqdm(data_loader, unit="batch") as data_loader:
                for batch in data_loader:
                    batch = [v.to(self.device) for v in batch]

                    out = model(batch)['logits'].detach().cpu().numpy()
                    predictions.append(out)

        return predictions


class Trainer():
    def __init__(self, epochs, device, transform):
        self.epochs = epochs
        self.device = device
        self.distance = distances.DotProductSimilarity()
        self.loss_fnc = losses.ContrastiveLoss(pos_margin=1, neg_margin=0, distance=distances.DotProductSimilarity())
        self.transform = transform

    def fit(self, model, data_loader, optimizer, lr_scheduler):
        best_loss = 10
        for epoch in range(1, self.epochs + 1):
            print('Epoch {}'.format(epoch))
            model.train()
            running_loss = 0.0
            with tqdm(data_loader, unit="batch") as data_loader:
                for batch in data_loader:
                    batch_size = batch[0].size()[0]
                    batch = torch.cat([batch[0], batch[1]])
                    batch = [list(b) for b in list(batch)]
                    batch = self.transform(batch, return_tensors="pt")
                    batch.to(self.device)

                    sequence_output = model(**batch)[0]
                    logits = sequence_output[:, 0]

                    query = logits[:batch_size]
                    positive_key = logits[batch_size:int(batch_size*2)]
                    # negative_key = logits[int(batch_size * 2):int(batch_size * 3)]

                    # p = torch.arange(batch_size).to(self.device)
                    # n = torch.arange(batch_size, int(batch_size * 2)).to(self.device)
                    # indices_tuple = (p,p,p,n)

                    label = torch.arange(batch_size).to(self.device)

                    loss = self.loss_fnc(query, labels=label, ref_emb=positive_key, ref_labels=label)

                    optimizer.zero_grad()
                    loss.backward()
                    lr_scheduler.step()
                    optimizer.step()

                    running_loss += loss.item()
                    data_loader.set_postfix(loss=loss.item())
            avg_loss = running_loss / len(data_loader)
            print('Loss {}'.format(avg_loss))

            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_pretrained('./ckpts/.')

    def validate(self, model, data_loader):
        model.to(self.device)
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            with tqdm(data_loader, unit="batch") as data_loader:
                for batch in data_loader:
                    batch_size = batch[0].size()[0]
                    batch = torch.cat([batch[0], batch[1], batch[2]])
                    batch = [list(b) for b in list(batch)]
                    batch = self.transform(batch, return_tensors="pt")
                    batch.to(self.device)

                    sequence_output = model(**batch)[0]
                    logits = sequence_output[:, 0]

                    query = logits[:batch_size]
                    positive_key = logits[batch_size:int(batch_size*2)]
                    # negative_key = logits[int(batch_size * 2):int(batch_size * 3)]

                    label = torch.arange(batch_size).to(self.device)

                    loss = self.loss_fnc(query, labels=label, ref_emb=positive_key, ref_labels=label)

                    # p = torch.arange(batch_size).to(self.device)
                    # n = torch.arange(batch_size, int(batch_size * 2)).to(self.device)
                    # indices_tuple = (p,p,p,n)

                    # loss = self.loss_fnc(query, indices_tuple=indices_tuple,
                    #                      ref_emb=torch.cat([positive_key, negative_key]))
                    running_loss += loss.item()
                    data_loader.set_postfix(loss=loss.item())
        avg_loss = running_loss / len(data_loader)
        print('val Loss {}'.format(avg_loss))

    def predict(self, model, data_loader):
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            with tqdm(data_loader, unit="batch") as data_loader:
                for batch in data_loader:
                    batch_size = batch[0].size()[0]
                    batch = torch.cat([batch[0], batch[1], batch[2]])
                    batch = [list(b) for b in list(batch)]
                    batch = self.transform(batch, return_tensors="pt")
                    batch.to(self.device)

                    sequence_output = model(**batch)[0]
                    logits = sequence_output[:, 0]

                    query = logits[:batch_size]
                    positive_key = logits[batch_size:int(batch_size*2)]
                    # negative_key = logits[int(batch_size * 2):int(batch_size * 3)]

                    dist_pos = self.distance(query[0].unsqueeze(0), positive_key[0].unsqueeze(0)).detach().cpu().numpy()
                    dist_neg1 = self.distance(query[0].unsqueeze(0), positive_key[1].unsqueeze(0)).detach().cpu().numpy()
                    dist_neg2 = self.distance(query[0].unsqueeze(0), positive_key[2].unsqueeze(0)).detach().cpu().numpy()
                    dist_neg3 = self.distance(query[0].unsqueeze(0), positive_key[3].unsqueeze(0)).detach().cpu().numpy()

                    print(dist_pos, dist_neg1, dist_neg2, dist_neg3)