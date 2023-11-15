def sanitize(value):
    tokens = ['<unk>', '</s>', '<pad>', '<pad>']
    result = value
    for token in tokens:
        result = result.replace(token, '')
    return result