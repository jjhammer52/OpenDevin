STOP_WORDS = ["Observation:", "Observation:\n"]

def convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=True):
    # Basic mock version
    system_msg = next((m for m in messages if m["role"] == "system"), None)
    user_msgs = [m for m in messages if m["role"] == "user"]
    tool_descriptions = "\n\n".join([f"{tool['function']['name']}: {tool['function']['description']}" for tool in tools])

    new_messages = []
    if system_msg:
        new_messages.append(system_msg)
    if add_in_context_learning_example:
        new_messages.append({
            "role": "system",
            "content": f"You have access to the following tools:\n{tool_descriptions}"
        })
    new_messages.extend(user_msgs)
    return new_messages

def convert_non_fncall_messages_to_fncall_messages(messages, tools):
    # Naive mock-up, assumes last message is assistant output to be rewritten
    last = messages[-1]
    if "content" not in last or not isinstance(last["content"], str):
        return messages  # nothing to do

    # Insert fake tool call metadata
    messages[-1] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call-1",
            "function": {
                "name": tools[0]['function']['name'],
                "arguments": "{}"
            },
            "type": "function"
        }]
    }
    return messages
