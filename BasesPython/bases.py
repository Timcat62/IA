notes = [10, 11, 15, 9, 7]
somme = 0
for i in range(len(notes)):
    somme += notes[i]
moyenne = somme / len(notes)
if moyenne >= 10:
    print("Admis")
else:
    print("Rattrapage nécessaire...")