# Initialize a dictionary
CountDictionary = {}
# Get Input from User
MyStringInp = input("Please enter you string")
MyString = ''.join(MyStringInp.split())
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
checker = 0
for letter, frequency in CountDictionary.items():
    # Check if a letter has occurred only once and come out of loop
    if frequency == 1:
        print("first non-repeated characters in your string is ", letter)
        checker = 1
        break

if checker == 0:
    print("There are no non-repeated characters in your string")

