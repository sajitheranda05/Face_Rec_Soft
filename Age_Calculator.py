from datetime import datetime

class AgeCalculator:
    def __init__(self, nic_number):
        if nic_number is not None:
            self.nic_number = nic_number.replace(" ", "")
        else:
            self.nic_number = nic_number

    def calculate(self):
        if self.nic_number is not None:
            if len(self.nic_number) == 10:
                birth_year = int(self.nic_number[:2]) + 1900  
            elif len(self.nic_number) == 12:
                birth_year = int(self.nic_number[:4])  
            else:
                raise ValueError("Invalid NIC number format. Must be either 9 or 12 digits.")

            current_year = datetime.now().year
            age = current_year - birth_year
        
        else:
            age = None


        return age

