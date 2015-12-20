
def _nn_distances_bf(points):
    """
    Brute force nearest neighbors

    """
    n = len(points)
    d_mins = [MAXD] * n
    neighbors = [-1] * n
    for i, point_i in enumerate(points[:-1]):
        i_x, i_y = point_i
        for j in range(i+1, n):
            point_j = points[j]
            j_x, j_y = point_j
            dx = i_x - j_x
            dy = i_y - j_y
            d_ij = dx*dx + dy*dy
            if d_ij < d_mins[i]:
                d_mins[i] = d_ij
                neighbors[i] = j
            if d_ij < d_mins[j]:
                d_mins[j] = d_ij
                neighbors[j] = i
    d_mins = [math.sqrt(d_i) for d_i in d_mins]
    return neighbors, d_mins


def _d_min_bf(points):
    """
    Brute force mean nearest neighbor statistic

    """
    neighbors, d_mins = _nn_distances_bf(points)
    n = len(d_mins)
    return sum(d_mins)/n


def _G_bf(points, k=10):
    """
    Brute force cumulative frequency distribution of nearest neighbor
    distances
    """
    neighbors, d_mins = _nn_distances_bf(points)

    d_max = max(d_mins)
    w = d_max/k
    n = len(d_mins)

    d = [w*i for i in range(k+2)]
    cdf = [0] * len(d)
    for i, d_i in enumerate(d):
        smaller = [d_i_min for d_i_min in d_mins if d_i_min <= d_i]
        cdf[i] = len(smaller)*1./n
    return d, cdf


def _mbr_bf(points):
    """
    Minimum bounding rectangle, brute force
    """
    min_x = min_y = MAXD
    max_x = max_y = MIND
    for point in points:
        x, y = point
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y
    return min_x, min_y, max_x, max_y


def _F_bf(points, n=100):
    x0, y0, x1, y1 = _mbr_bf(points)
    ru = random.uniform
    r_points = [(ru(x0, x1), ru(y0, y1)) for i in xrange(n)]
    d_mins = [MAXD] * n
    neighbors = [-9] * n
    for i, r_point in enumerate(r_points):
        d_i = MAXD
        x0, y0 = r_point
        for j, point in enumerate(points):
            x1, y1 = point
            dx = x0-x1
            dy = y0-y1
            d = dx*dx + dy*dy
            if d < d_i:
                d_mins[i] = d
                neighbors[i] = j
                d_i = d
    return [math.sqrt(d_min_i) for d_min_i in d_mins], neighbors


def _F_cdf_bf(points, n=100, k=10,):
    d, g_cdf = _G_bf(points, k=k)
    d_mins, neighbors = _F_bf(points, n)
    cdf = [0] * len(d)
    for i, d_i in enumerate(d):
        smaller = [d_i_min for d_i_min in d_mins if d_i_min <= d_i]
        cdf[i] = len(smaller)*1./n
    return d, cdf


def _k_bf(points, n_bins=100):
    n = len(points)
    x0, y0, x1, y1 = _mbr_bf(points)
    d_max = (x1-x0)**2 + (y1-y0)**2
    d_max = math.sqrt(d_max)
    w = d_max / (n_bins-1)
    d = [w*i for i in range(n_bins)]
    ks = [0] * len(d)
    for i, p_i in enumerate(points[:-1]):
        x0, y0 = p_i
        for j in xrange(i+1, n):
            x1, y1 = points[j]
            dx = x1-x0
            dy = y1-y0
            dij = math.sqrt(dx*dx + dy*dy)
            uppers = [di for di in d if di >= dij]
            for upper in uppers:
                ki = d.index(upper)
                ks[ki] += 2
    return ks, d



