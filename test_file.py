lyrics_text = open("BROCKHAMPTON.txt", "r", encoding="utf-8").read()
#print(len(lyrics_text))

# Unique chars in lyrics file
vocabulary = list(sorted(set(lyrics_text))) 

print(len(vocabulary))