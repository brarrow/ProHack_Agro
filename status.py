

class Status(object):
    def __init__(self):
        self.persons = []
        self.persons_with_violation = []

    def check_status(self):
        for person in self.persons:
            if person.have_violation():
                self.persons_with_violation.append(person)
                print(person.get_info_str())

    def clear(self):
        self.persons = []
        self.persons_with_violation = []