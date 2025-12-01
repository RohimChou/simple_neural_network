class DosageData:
    def __init__(self, dosage, effectiveness):
        """
        Dto store dosage test data

        :param dosage: scalar, between 0 and 10 mg
        :type dosage: float
        :param effectiveness: between 0 and 1
        :type effectiveness: float
        """
        self.dosage = dosage
        self.effectiveness = effectiveness