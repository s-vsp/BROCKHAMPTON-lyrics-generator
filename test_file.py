import numpy as np


def get_title():

    lyrics_text = open("BROCKHAMPTON.txt", "r", encoding="utf-8").read()

    t = np.random.randint(1,len(lyrics_text),size=1)

    ignorable = [" ", "\n", ".", ",", "[", "]", "(", ")"]

    chars = []

    for i, c in enumerate(lyrics_text[t.item():t.item()+20]):
        chars.append(c)
        if c in ignorable and i == 0:
            continue
        elif c in ignorable and i != 0:
            chars.remove(c)
            break
    
    title = ""

    return print(title.join((chars)).upper())

get_title()
