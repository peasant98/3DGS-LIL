import pickle
import random
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import pandas as pd
import os

from gensim.models import KeyedVectors

from robot_control.segment_novel_view import GS, Segment3DGS


SYNONYMS_MOVE_TO = ["move to" ,"go to", "move to",
                    "head to", "travel to",
                    "go over to", "move over to",
                    "navigate to", "proceed to",
                    "approach", "advance to", "approach", "get to",
                    "go up to", "move close to", "move towards", "go towards",]

SYNONYMS_MOVE_AWAY = ["go away from", "move apart", "back away from",
                      "retreat from", "diverge from", "move away from", "get away from", "back off from", "move back from",
                      "move far from"]

class TextImageDataset(Dataset):
    def __init__(self, tokenizer_name='bert-base-uncased', config_gs_yml='outputs/panda-data/splatfacto/2024-06-05_151435/config.yml',
                 glove_path='glove.6B.50d.txt', text_embed_size=50):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations (image paths, text commands, EE velocities).
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
            tokenizer_name (string, optional): Name of the tokenizer to use for text commands.
        """
        self.data = []
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        
    ])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
        self.gs = GS(config_gs_yml)
        self.segmenter = Segment3DGS(self.gs)
        self.glove = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
        self.text_embed_size = text_embed_size
        
        self.synonyms_move_to = SYNONYMS_MOVE_TO
        self.synonyms_move_away = SYNONYMS_MOVE_AWAY
        
        random.shuffle(self.synonyms_move_to)
        random.shuffle(self.synonyms_move_away)
        
        self.train_move_to = self.synonyms_move_to[:int(len(self.synonyms_move_to) * 0.5)]
        self.test_move_to = self.synonyms_move_to[int(len(self.synonyms_move_to) * 0.5):]
        
        self.train_move_away = self.synonyms_move_away[:int(len(self.synonyms_move_away) * 0.5)]
        self.test_move_away = self.synonyms_move_away[int(len(self.synonyms_move_away) * 0.5):]
        
        # get others 
        
    
    def add_sample(self, text, item, image, velocity, pose):
        self.data.append((text, item, image, velocity, pose))

    def __len__(self):
        return len(self.data)
    
    def save(self, idx):
        # save to pickle file
        with open(f'full_exp_data{idx}.pkl', 'wb') as f:
            pickle.dump(self.data, f)
        self.data = []
        
    def read(self, idx):
        # read from pickle file
        with open(f'full_exp_data{idx}.pkl', 'rb') as f:
            data_arr = pickle.load(f)
            # combine two lists
            self.data = data_arr + self.data
        print(len(self.data ))
    
    def get_avg_gs_time(self):
        avg_time = self.segmenter.gs_model.get_avg_time()
        return avg_time
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        text, item, image, velocity, pose = self.data[idx]
        original_text = text
        
        train = True
        
        if len(self.data) < 5000:
            train = False
        if train and text == 'move to':
            text = random.choice(self.train_move_to)
        elif not train and text == 'move to':
            text = random.choice(self.test_move_to)
            
        if train and text == 'move away from':
            text = random.choice(self.train_move_away)
        elif not train and text == 'move away from':
            text = random.choice(self.test_move_away)
        position, quat = pose
        
        # if it's gs, get the novel view from pose
        rgb_gs, depth, mask = self.segmenter.segment_novel_view((position, quat), object_name=item, visualize=False, segment=True)
        
        mask = np.array(mask, dtype=np.uint8)
        # show rgb, depth, mask
        # if mask is not None:
        #     fig, ax = plt.subplots(1, 3)
        #     ax[0].imshow(rgb_gs)
        #     ax[1].imshow(depth)
        #     ax[2].imshow(mask)
        #     plt.show()
        
        if idx == 5:
            print(self.get_avg_gs_time(), "Average time for GS")
        
        text_tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        # Get the embeddings
        with torch.no_grad():
            outputs = self.model(**text_tokens)
        phrase_embeddings = outputs.last_hidden_state.mean(dim=1)
        # text_tokens = torch.tensor(text_tokens['input_ids'][0])
        
        # convert image to PIL image
        # resize depth to 224 x 224
        depth = cv2.resize(depth, (224, 224))
        # convert to tensor
        depth = torch.tensor(depth) 
        
        # add 1 dim to depth
        depth = depth.unsqueeze(0)
        
        mask = cv2.resize(mask, (224, 224))
        mask = torch.tensor(mask, dtype=torch.float32)
        mask = mask.unsqueeze(0)


        # use GS
        if self.transform:
            image = self.transform(rgb_gs)
            # image = self.transform(image)
            
        velocity = np.array(velocity)
        velocity = torch.tensor(velocity, dtype=torch.float32)
        
        return phrase_embeddings, image, mask, depth, velocity, original_text, text, item

if __name__ == "__main__":
    config_path = 'outputs/panda-data/splatfacto/2024-06-05_151435/config.yml'
    text_embed_size = 50
    dataset = TextImageDataset(config_gs_yml=config_path, text_embed_size=text_embed_size)
    
    for i in range(20):
        dataset.read(i)
    
    # random idxes in dataset
    idxes = np.random.choice(len(dataset), 5)
    for idx in idxes:
        text, image, velocity = dataset[idx]
