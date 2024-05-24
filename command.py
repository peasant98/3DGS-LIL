# OpenAI API: Chat example

from openai import OpenAI

if __name__ == "__main__":
    client = OpenAI()
    
    phrase = "pick up the xbox controller"

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a word smith. You have been given a phrase and you must identify the action and object. Always give an answer."},
        {"role": "user", "content": f"I have the phrase '{phrase}'. What is the action, and what is the object? Remove all articles of speech.."}
    ]
    )
    
    parsed_message = completion.choices[0].message.content.split('\n')
    action_msg = parsed_message[0]
    object_msg = parsed_message[1]
    
    action_msg = action_msg.split(": ")[-1].strip()
    
    object_ = object_msg.split(": ")[-1].strip()
    
    print('Action to embed into Consistency Policy: ', action_msg)
    print('Object in question: ', object_)