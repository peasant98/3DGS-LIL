
import os, time, cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from ai_controller import UnifiedAIController
from robo_dataset import TextImageDataset


def plot_image_with_velocity(image, velocity, sample_index):
    # Convert the image tensor to a NumPy array and transpose the dimensions to HxWxC
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    velocity_np = velocity.cpu().numpy()
    velocity_np = -velocity_np  # Invert the velocity vector
    plt.figure()
    plt.imshow(image_np)
    
    # Define the starting point (center of the image)
    start_point = (image_np.shape[1] // 2, image_np.shape[0] // 2)
    
    # Define the end point by adding the velocity vector to the start point
    end_point = (start_point[0] + int(velocity_np[0] * 100), 
                 start_point[1] + int(velocity_np[1] * 100))
    
    # swap end and start points
    # Plot the velocity vector
    plt.arrow(start_point[0], start_point[1], 
              end_point[0] - start_point[0], end_point[1] - start_point[1], 
              head_width=10, head_length=15, fc='red', ec='red')
    
    plt.title(f'Sample {sample_index} - Velocity: {velocity_np}')
    plt.show()

def train(dataset, embed_size=50):
    # train dataset on the collected data
    print("Beginning Training")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # have an 0.8 train, 0.2 test split
    small_train_size = int(0.8 * len(train_dataset))
    train_dataset, _ = random_split(train_dataset, [small_train_size, len(train_dataset) - small_train_size])
    
    small_train_size1 =  int(0.8 * len(test_dataset))
    test_dataset, _ = random_split(test_dataset, [small_train_size1, len(test_dataset) - small_train_size1])
    
    print("Train size: ", len(train_dataset))
    print("Test size: ", len(test_dataset))
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = UnifiedAIController(embed_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Device: ", device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        correct_direction_num = 0
        total = 0
        samples = 0
        
        for text_embedding, images, masks, depths, velocities, text, transformed_text, item in train_loader:
            samples += batch_size
            print(f"Sample: {samples}")
            optimizer.zero_grad()
            
            # plot images
            images = images.to(device)
            velocities = velocities.to(device)  
            depths = depths.to(device)
            masks = masks.to(device)
            
            # plot image, mask, velocity, text
            # for i in range(len(images)):
            #     print(text[i], item[i])
            #     plot_image_with_velocity(images[i], velocities[i], i)
            # plot_image_with_velocity(first_img, first_vel, 0)
     
            
            # combine image, depth, mask
            combined = torch.cat((images, depths, masks), dim=1)
            
            text_embedding = text_embedding.to(device)
            
            
            # Forward pass
            outputs, model_text_embedding = model(combined, text_embedding)
            
            loss = criterion(outputs, velocities)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            signs_match = torch.sign(outputs) == torch.sign(velocities)
            all_signs_match = torch.all(signs_match)
            count = sum((row == torch.tensor([True, True, True], device='cuda:0')).all().item() for row in signs_match)
            correct_direction_num += count
            total += len(velocities)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        print(f"Total train correct: {correct_direction_num / total * 100}%")

        model.eval()
        test_loss = 0.0
        correct_direction_num = 0
        total = 0
        avg_cosine_sim = 0
        with torch.no_grad():
            for sample_index, (text_embedding, images, masks, depths, velocities, text, transformed_text, item) in enumerate(test_loader):
                images = images.to(device)
                depths = depths.to(device)
                velocities = velocities.to(device)
                masks = masks.to(device)
                
                combined = torch.cat((images, depths, masks), dim=1)
                
                text_embedding  = text_embedding.to(device)
                
                outputs, model_text_embedding = model(combined, text_embedding)
                
                # pca = PCA(n_components=2)
                # king_embedding_np = model_text_embedding.cpu().numpy()
                # reduced_embeddings = pca.fit_transform(king_embedding_np)

                # # Plot in 2D using Matplotlib
                # plt.rcParams.update({'font.size': 20, 'axes.titlesize': 20, 'axes.labelsize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20})
                # plt.figure(figsize=(10, 8))
                # unique_labels = list(set(text))
                # colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

                # for label, color in zip(unique_labels, colors):
                #     idx = [i for i, l in enumerate(text) if l == label]
                #     plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=label, color=color, s=100, alpha=0.7, edgecolors='w')
                    
                # for i in range(len(transformed_text)):
                #     plt.annotate(transformed_text[i], (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=14, alpha=0.75)

                # plt.title('PCA of Movement in 2D')
                # plt.xlabel('PCA Component 1')
                # plt.ylabel('PCA Component 2')
                # plt.legend()
                # plt.grid(True)
                # plt.show()
                
                # plt.rcParams.update({'font.size': 20, 'axes.titlesize': 20, 'axes.labelsize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20})
                # plt.figure(figsize=(10, 8))
                # unique_labels = list(set(text))
                # colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

                # for label, color in zip(unique_labels, colors):
                #     idx = [i for i, l in enumerate(text) if l == label]
                #     plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=label, color=color, s=100, alpha=0.7, edgecolors='w')
                    
                # for i in range(len(transformed_text)):
                #     plt.annotate(transformed_text[i], (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=16, alpha=0.75)

                # plt.title('PCA of Movement in 2D')
                # plt.xlabel('PCA Component 1')
                # plt.ylabel('PCA Component 2')
                # plt.legend()
                # plt.grid(True)
                # plt.show()
                
                loss = criterion(outputs, velocities)
                test_loss += loss.item() * images.size(0)
                
                # Plot and save each image with the predicted velocity overlaid
                for i in range(images.size(0)):
                    # see if signs match 
                    # compute cosine similarity between predicted and actual velocity
                    cosine_similarity = nn.CosineSimilarity(dim=0)
                    cosine_sim = cosine_similarity(outputs[i], velocities[i])
                    
                    avg_cosine_sim += cosine_sim.item()
                    signs_match = torch.sign(outputs[i]) == torch.sign(velocities[i])
                    all_signs_match = torch.all(signs_match)
                    if all_signs_match:
                        correct_direction_num +=1
                    total += 1
                    # print(f"Sample {sample_index * len(images) + i} - All signs match: {all_signs_match}")
                    # print(text)
                    # plot_image_with_velocity(images[i], outputs[i], sample_index * len(images) + i)
                        
        avg_cosine_sim = avg_cosine_sim / len(test_loader.dataset)
        print(f"Avg cosine similarity: {avg_cosine_sim:.4f}")
        test_loss = test_loss / len(test_loader.dataset)
        print(f"Test Loss: {test_loss:.4f}")
        total_correct = correct_direction_num / total
        print(f"Total test correct: {total_correct * 100}%")
        
    
    print("Finished Training")
    # save model
    torch.save(model.state_dict(), 'final_model.pth')
    
    
    
if __name__ == '__main__':
    config_path = 'outputs/panda-data/splatfacto/2024-06-05_151435/config.yml'
    text_embed_size = 50
    dataset = TextImageDataset(config_gs_yml=config_path, text_embed_size=text_embed_size)
    
    for i in range(1, 25):
        dataset.read(i)
    
    train(dataset, embed_size=text_embed_size)