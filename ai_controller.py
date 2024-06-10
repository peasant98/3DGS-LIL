import torch
import torch.nn as nn
import torchvision.models as models

class UnifiedAIController(nn.Module):
    def __init__(self, embed_size):
        super(UnifiedAIController, self).__init__()
        # Unified feature extraction using ResNet18
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Adapting conv1 layer to accept 5-channel input (3 for RGB, 1 for depth, 1 for mask)
        self.resnet.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        bert_embed_size = 768
        self.text_fc = nn.Sequential(
            nn.Linear(bert_embed_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Combine image features and phrase embedding
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Output: 3D velocity
        )

    def forward(self, x_combined, text_embedding):
        # Ensure input is of type torch.cuda.FloatTensor
        x_combined = x_combined.float()
        text_embedding = text_embedding.float()

        # Get features from combined input (RGB + depth + mask)
        combined_features = self.resnet(x_combined)
        
        # Process text embedding
        text_embedding = text_embedding.squeeze(1)
        processed_text_embedding = self.text_fc(text_embedding)
        
        # Concatenate combined features and processed text embedding
        combined_features = torch.cat((combined_features, processed_text_embedding), dim=1)
        
        # Pass combined features through the fully connected layers
        output = self.fc(combined_features)
        
        return output, processed_text_embedding

def main():
    # Example input dimensions
    batch_size = 8
    channels = 5  # 3 for RGB, 1 for depth, 1 for mask
    height = 224
    width = 224
    embed_size = 768

    # Create dummy data for combined input and text embedding
    x_combined = torch.randn(batch_size, channels, height, width)
    text_embedding = torch.randn(batch_size, 1, embed_size)

    # Instantiate the model
    model = UnifiedAIController(embed_size)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        x_combined = x_combined.cuda()
        text_embedding = text_embedding.cuda()

    # Perform a forward pass
    output = model(x_combined, text_embedding)

    # Print the output
    print("Model output:", output.shape)

if __name__ == "__main__":
    main()
