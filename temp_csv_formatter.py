import csv

if __name__=="__main__":
    unformatted_file = csv.reader(open('EURUSD30.csv', 'r'))
    new_file = open('EURUSD30_formatted.csv', 'w')
    new_file.write("DateTime,Close\n")
    for line in unformatted_file:
        line = line[0].split("\t")
        new_file.write(f"{line[0]}, {line[4]}\n")
    unformatted_file = None
    new_file = None