# Initialize a dictionary
CountDictionary = {}
# Get Input from User
MyString = input("Please enter you string")
for letter in MyString:
    # If letter is repeating
    if letter in CountDictionary:
        # increment its count
        count = CountDictionary.get(letter)+1
        # Update that letter's count to +1
        CountDictionary.update({letter: count})
    else:
        # word's first time occurrence
        CountDictionary[letter] = 1
for letter, frequency in CountDictionary.items():
    # Check if a letter has occurred only once and come out of loop
    if frequency == 1:
        print("first non-repeated characters in you string is ", letter)
        break

