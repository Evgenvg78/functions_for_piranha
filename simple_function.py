import csv

def simple_csv():
    print("Hello from GitHub with CSV!")
    print('sssuukkaa')
    data = [["Name", "Age"], ["John", "23"], ["Anna", "34"]]
    print('sssssssss')
    with open("output.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("CSV file created")
