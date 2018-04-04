def readClasses(fileName):
    '''(string) -> dict
        Returns a dictionary with keys as monkey labels, value as a dictionary
        containing the  monkey's Latin Name, Common Name, number of training images
        and number of validation images.
    '''
    retDict = {}
    with open(fileName,'r') as f:
        line = f.readline().replace('\t',"").strip()
        col_labels = [x.strip() for x in line.split(',')]
        while line:    
            line = f.readline().replace('\t',"").strip()
            items = [x.strip() for x in line.split(',')]
            monk = dict(zip(col_labels,items))
            retDict[ monk[col_labels[0]] ] = monk
    return retDict

if __name__ == "__main__":
    classDict = readClasses("monkey_labels.txt")
    for i,j in classDict.items(): print("{} {} ".format(i, j))
