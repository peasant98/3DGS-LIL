# OpenAI API: Chat example

import time
import numpy as np
import torch
from openai import OpenAI
import torch.nn as nn

actions = [
    "move", "move to", "move close to", "push",  
    "pull", "set up", "carry", "pick up", "place", "throw", 
    "roll", "slide", "move away from", "push away", 
    "grab", "carry to", "lift", "drop", "set down", "go to",
    "travel to", "go away from"
]

objects = [
    "chair", "table", "desk", "sofa", "lamp", "book", "cup", 
    "bottle", "box", "pen", "phone", "laptop", "notebook", 
    "pillow", "blanket", "backpack", "shoe", "shirt", "hat", 
    "key", "watch", "remote", "mouse", "keyboard", "bag"
]


class OpenAICommander():
    def __init__(self):
        self.client = OpenAI()
    
    def get_action_and_object(self, phrase):
        # content_to_send = f"I have the phrase '{phrase}'. What is the action, and what is the object? If the phrase includes the word 'to' AFTER the first word, please include it in the action phrase."
        content_to_send = f"I have the phrase '{phrase}'. What is the action, and what is the object?"

        completion = self.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a word smith. You have been given a phrase and you must identify the full action phrase and object. Give the answer in the format 'action: <action>,\n object: <object>'."},
            {"role": "user", "content": f"{content_to_send}"}
        ]
        )
        parsed_message = completion.choices[0].message.content.split('\n')
        if len(parsed_message) == 1:
            # split with comma
            parsed_message = parsed_message[0].split(",")
        action_msg = parsed_message[0]
        object_msg = parsed_message[1]
        action_msg = action_msg.split(": ")[-1].strip()

        object_ = object_msg.split(": ")[-1].strip()
        
        # get rid of commas
        object_ = object_.replace(",", "")
        action_msg = action_msg.replace(",", "")

        return action_msg, object_

def run_simple_experiment():
    commander = OpenAICommander()
    loss_fn = nn.CrossEntropyLoss()
    examples = []
    for action in actions:
        for obj in objects:
            if len(examples) >= 500:
                break
            examples.append({"action_phrase": action, "object": obj, "full_phrase": f"{action} the {obj}"})
    action_losses = []
    object_losses = []
    print(len(actions), len(objects), len(examples))
    for idx, example in enumerate(examples):
        action, object_ = commander.get_action_and_object(example['full_phrase'])
        
        object_ = object_.split(" ")[-1]
        object_ = object_.replace(",", "")
        object_ = object_.replace(".", "")
        # remove ' mark from object
        # decapitalize
        object_ = object_.lower()
        object_ = object_.replace("'", "")
        action = action.lower()
        
        if "n/a" in object_: # If the object is not applicable, skip this example
            action, object_ = commander.get_action_and_object(example['full_phrase'])
        
            object_ = object_.split(" ")[-1]
            object_ = object_.replace(",", "")
            object_ = object_.replace(".", "")
            # remove ' mark from object
            # decapitalize
            object_ = object_.lower()
            object_ = object_.replace("'", "")
            
            action = action.lower()

        full_phrase = example['full_phrase']

        print("Full Phrase: ", full_phrase)
        print("Predicted Action: ", action)
        print("Predicted Object: ", object_)

        # Get the length of the phrase and split it into words
        phrase_length = len(full_phrase.split(" "))
        full_phrase_words = full_phrase.split(" ")

        # Construct logits for the full phrase
        object_logits = torch.zeros((1, phrase_length), dtype=torch.float32)
        object_logits = torch.full((1, phrase_length), -1.0)
        try:
            object_idx = full_phrase_words.index(object_)
            object_logits[0, object_idx] = 1.0  # Use a high positive value to indicate high confidence
        except ValueError:
            pass
        # Ground truth indices for the object (allowing multiple correct outputs)
        gt_object_idx = full_phrase_words.index(object_)
        gt_object = torch.zeros((1, phrase_length), dtype=torch.float32)
        gt_object[0, gt_object_idx] = 1.0  # Binary target

        # Define the loss function
        loss_fn = nn.BCEWithLogitsLoss()

        # Compute the object loss
        loss_object = loss_fn(object_logits, gt_object)

        # Get action loss
        action_logits = torch.full((1, phrase_length), -1.0, dtype=torch.float32) 
        gt_action_phrase = example['action_phrase']
        
        # Convert action_logits to a list to allow dynamic resizing
        action_logits_list = action_logits.tolist()[0]

        for word in action.split(" "):
            try:
                # Remove punctuation
                word = word.replace(",", "")
                word = word.replace(".", "")
                action_idx = full_phrase_words.index(word)
                action_logits_list[action_idx] = 1.0  # High positive value to indicate high confidence
            except ValueError:
                # Append high positive value to action logits list to handle the new word
                action_logits_list.append(1.0)

        # Convert the updated list back to a tensor
        action_logits = torch.tensor([action_logits_list], dtype=torch.float32)

        # Ground truth indices for the action (allowing multiple correct outputs)
        gt_action_indices = [full_phrase_words.index(word) for word in gt_action_phrase.split(" ")]
        gt_action = torch.zeros((1, len(action_logits_list)), dtype=torch.float32)
        for idx in gt_action_indices:
            gt_action[0, idx] = 1.0  # Binary target

        # Compute the action loss
        loss_action = loss_fn(action_logits, gt_action)

        # print("Object Logits: ", object_logits)
        # print("GT Object: ", gt_object)
        print("Object Loss: ", loss_object.item())
        # print("Action Logits: ", action_logits)
        # print("GT Action: ", gt_action)
        print("Action Loss: ", loss_action.item())
        
        action_losses.append(loss_action.item())
        object_losses.append(loss_object.item())
        time.sleep(0.5)
        
        
    print("Average Action Loss: ", np.mean(action_losses))
    print("Average Object Loss: ", np.mean(object_losses))
    
    # print("Phrase: ", phrase)
    # print('Action to embed into Policy: ', action)
    # print('Object in question: ', object_)

if __name__ == "__main__":
    commander = OpenAICommander()
    phrase = "move to the chair"
    
    action, object_ = commander.get_action_and_object(phrase)
    
    print("Phrase: ", phrase)
    print('Action to embed into Policy: ', action)
    print('Object in question: ', object_)
    
    run_simple_experiment()