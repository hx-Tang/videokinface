import torch
from torch.utils.data.dataloader import DataLoader
from trainer import PreTrainer, Trainer
from dataset import Contrastive_pretrain
import numpy as np
from transformers import VideoMAEModel, VideoMAEFeatureExtractor, VideoMAEConfig, AutoFeatureExtractor
from transformers import TrainingArguments, get_scheduler
data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'

device = torch.device('cuda')

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 10,
    "log_level": "error",
    "report_to": "none",
}

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    fp16=True,
    learning_rate=5e-5,
    **default_args,
)

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base", size=112)


def pre_train():

    train_label = np.load('./train.npy')
    train_set = Contrastive_pretrain(data_path, train_label)
    train_data = DataLoader(
        train_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=4)

    config = VideoMAEConfig(image_size=112, num_frames=5)
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base", config=config)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=training_args.learning_rate)
    num_training_steps = training_args.num_train_epochs * len(train_data)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=1,
                                 num_training_steps=num_training_steps)

    trainer = PreTrainer(training_args.num_train_epochs, device, feature_extractor)
    trainer.fit(model, train_data, optimizer, lr_scheduler)


def pre_validate():
    test_label = np.load('./test.npy')
    test_set = Contrastive_pretrain(data_path, test_label)
    test_data = DataLoader(
        test_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=4)

    config = VideoMAEConfig(image_size=112, num_frames=5)
    model = VideoMAEModel.from_pretrained("C:\\Users\\13661\\PycharmProjects\\KinshipVerification\\tmp", config=config)
    model.to(device)

    trainer = PreTrainer(training_args.num_train_epochs, device, feature_extractor)
    trainer.validate(model, test_data)


def train():

    train_label = np.load('./train.npy')
    train_set = Contrastive_pretrain(data_path, train_label)
    train_data = DataLoader(
        train_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=4)

    config = VideoMAEConfig(image_size=112, num_frames=5)
    model = VideoMAEModel.from_pretrained("C:\\Users\\13661\\PycharmProjects\\KinshipVerification\\tmp", config=config)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=training_args.learning_rate)
    num_training_steps = training_args.num_train_epochs * len(train_data)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=1,
                                 num_training_steps=num_training_steps)

    trainer = Trainer(training_args.num_train_epochs, device, feature_extractor)
    trainer.fit(model, train_data, optimizer, lr_scheduler)


def validate():
    test_label = np.load('./test.npy')
    test_set = Contrastive_pretrain(data_path, test_label)
    test_data = DataLoader(
        test_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=4)

    config = VideoMAEConfig(image_size=112, num_frames=5)
    model = VideoMAEModel.from_pretrained("C:\\Users\\13661\\PycharmProjects\\KinshipVerification\\ckpts", config=config)
    model.to(device)

    trainer = Trainer(training_args.num_train_epochs, device, feature_extractor)
    trainer.validate(model, test_data)


def test():
    test_label = np.load('./test.npy')
    test_set = Contrastive_pretrain(data_path, test_label)
    test_data = DataLoader(
        test_set, batch_size=4, shuffle=True, num_workers=4)

    config = VideoMAEConfig(image_size=112, num_frames=5)
    model = VideoMAEModel.from_pretrained("C:\\Users\\13661\\PycharmProjects\\KinshipVerification\\ckpts", config=config)
    model.to(device)

    trainer = Trainer(training_args.num_train_epochs, device, feature_extractor)
    trainer.predict(model, test_data)


if __name__ == "__main__":
    # pre_train()
    # pre_validate()
    train()
    # validate()
    # test()