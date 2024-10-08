import csv

def simple_hello():
    print("Привет Это функция работает:)!")

def cre_csv():
    print("Hello from GitHub with CSV!")
    data = [["Name", "Age"], ["John", "23"], ["Anna", "34"]]
    with open("output.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("CSV file created")
