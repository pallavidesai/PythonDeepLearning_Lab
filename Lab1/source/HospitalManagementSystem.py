# Hospital management system
# classes- patient, doctor, employee, person, student, medicalintern student
# declare a patient class


class Patient():
    # defines constructor with parameters name, number, age, day
    def __init__(self, name, patient_number, age, day):
        self.fullname = name
        self.patient_number = patient_number
        self.age = age
        self.day = day
    # prints name age and number of person

    def get_name(self):
        print("Patient Name:", self.fullname)


    def get_number(self):
        print("Patient Number:", self.patient_number)

    def get_age(self):
        print("age:", self.age)

# defining a private member

    def __getday_(self):
        print("Day admitted:", self.day)

# define class doctor


class Doctor():
    def __init__(self, dname, department):
        self.dname = dname
        self.dep = department

    # printing name and department of doctor
    def get_dname(self):
        print("Name of the doctor:", self.dname)

    def get_department(self):
        print("Department name:", self.dep)

class Person():

    def __init__(self, firstname, lastname):
        self.fname = firstname
        self.lname = lastname

    def get_firstname(self):
        print("first name of medical intern:", self.fname)

    def get_lastname(self):
        print("last name of medical intern:", self.lname)


# single inheritance and one super call
class Employee(Person):
    def __init__(self, employeeid, firstname, lastname):
        self.empid = employeeid
        super().__init__(firstname, lastname)
    def get_employeeid(self):
        print("medical intern student id:", self.empid)


class Student:
    def __init__(self, year):
        self.gyear = year

    def get_year(self):
        print("grad year:", self.gyear)

# using multiple inheritance
class MedicalInternStudent(Student, Employee):

    def __init__(self, year, employeeid, firstname, lastname):
        Student.__init__(self,year)
        Employee.__init__(self, employeeid, firstname, lastname)


a =  Patient("mihir", 15, 23, "Thursday")
b = Doctor("pallavi", "surgery")
c = MedicalInternStudent("third year", 23, "nick", "jonas")

a.get_name()
a.get_age()
a.get_number()
print("---------------------------------")
b.get_dname()
b.get_department()
print("---------------------------------")
c.get_employeeid()
c.get_year()
c.get_firstname()
c.get_lastname()

