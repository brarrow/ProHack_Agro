

class Person(object):
    def __init__(self, img=None, name="", H=True, P=True, G=True):
        self.img = img
        self.name = name
        self.H = H
        self.P = P
        self.G = G

    def have_violation(self):
        return not (self.H and self.P and self.G)

    def get_info_str(self):
        return "Name: {}, Helmet: {}, Protivagas: {}, Glasses: {}".format(self.name, self.H, self.P, self.G)