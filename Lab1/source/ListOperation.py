# Code to print list of students who are attending
#  “python” classes but not “Web Application”
# Take inout from user and put it in set
PythonInput = input("Enter python student names separated by comma")
PythonStudents = {x for x in PythonInput.split(",")}
# Take inout from user and put it in set
WebInput = input("Enter WebApplication student names separated by comma")
WebApplicationStudents = {x for x in WebInput.split(",")}
# .difference function  will give set of only python students
OnlyPythonStudents= PythonStudents.difference(WebApplicationStudents)
# Print Out the set
print("Students who are attending only python classes are",OnlyPythonStudents)