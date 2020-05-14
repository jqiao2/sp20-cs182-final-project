def get_labels():
    f = open('data/tiny-imagenet-200/val/val_annotations.txt', 'r')
    labels = {}
    for line in f:
        j = line.split('\t')
        k = j[0].split('.')[0].split('_')[1]
        labels[int(k)] = j[1]
    return labels

def get_predictions(preds='eval_classified.csv'):
    f = open(preds)
    predictions = {}
    for line in f:
        j = line.split(',')
        predictions[int(j[0])] = j[1][:-1]
    return predictions

def accuracy(preds='eval_classified.csv'):
    c = 0
    p, l = get_labels(), get_predictions(preds)
    for k in p:
        if p[k] == l[k]:
            c += 1
    return c / 10000.

if __name__ == '__main__':
    print(accuracy() * 100, "%")
