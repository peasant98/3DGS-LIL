# OpenAI API: Chat example

from openai import OpenAI


class OpenAICommander():
    def __init__(self):
        self.client = OpenAI()
    
    def get_action_and_object(self, phrase):
        content_to_send = f"I have the phrase '{phrase}'. What is the action, and what is the object? Remove all articles of speech."

        completion = self.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a word smith. You have been given a phrase and you must identify the action and object. Always give an answer."},
            {"role": "user", "content": f"{content_to_send}"}
        ]
        )

        parsed_message = completion.choices[0].message.content.split('\n')
        action_msg = parsed_message[0]
        object_msg = parsed_message[1]

        action_msg = action_msg.split(": ")[-1].strip()

        object_ = object_msg.split(": ")[-1].strip()

        return action_msg, object_



if __name__ == "__main__":
    commander = OpenAICommander()
    phrase = "pick up the xbox controller"
    
    action, object_ = commander.get_action_and_object(phrase)
    
    print("Phrase: ", phrase)
    print('Action to embed into Policy: ', action)
    print('Object in question: ', object_)