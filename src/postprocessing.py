def countValue(l):
    d = {-1:0, 1:0}
    for v in l:
        # d[v] = d.get(v, 0) + 1
        d[v] += 1
    return d

def windows_post(y, size=1, threshold=2):
    """
    Convolue sur la liste et regarde le nombre de voisin dans la fenêtre centré sur le point. 

    * Même nombre de valeur des deux cotés = frontière = ne pas changer la valeur
    * 
    """
    new_y = y.copy()
    for i in range(size, len(y) - size):
        window = y[i - size : i + size + 1]
        d = countValue(window)
        if d[-1] < d[1]:
            new_y[i] = 1
        elif d[-1] > d[1]:
            new_y[i] = -1
        else:
            pass
    return new_y