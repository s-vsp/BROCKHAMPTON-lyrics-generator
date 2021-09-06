import numpy as np

lyrics_text = open("BROCKHAMPTON.txt", "r", encoding="utf-8").read()
#print(len(lyrics_text))

# Unique chars in lyrics file
vocabulary = list(sorted(set(lyrics_text))) 

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

    return title.join((chars)).upper()

print(get_title())


