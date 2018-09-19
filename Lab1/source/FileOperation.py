# Code to remove everything in the File1 which is inside File2.
# Open two files
file1 = open("file1.txt", "r")
file2 = open("file2.txt", "r")
# Put contents of 2 files in string
str1 = file1.read()
str2 = file2.read()
# Convert both strings to lower case to make it case sensitive
str1=str1.lower()
str2=str2.lower()
# Close both files since they no longer are necessary
file1.close()
file2.close()
# Split contents of both files with white space
split_str1 = str1.split()
split_str2 = str2.split()
# Get common words in both strings using and operation of set
common_words = set(split_str1) & set(split_str2)
print(common_words)
# Clear Contents of file1 and open it in write mode
open("file1.txt", "w").close()
file1 = open("file1.txt", "r+")
# Remove words that are common with both strings
for word in split_str1:
    if not word in common_words:
        print(word)
        # Write those words to file along with a white space
        file1.write(word)
        file1.write(" ")

