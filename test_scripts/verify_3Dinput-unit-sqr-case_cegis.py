from tqdm import tqdm

mappings = [[[0.0, 0.0001, 0.0001], [1.0, 1.0, 1.0], 0], [[0.0001, 0.0002, 0.0], [1.0, 1.0, 0.0], 0], [[0.0001, 0.0002, -1.0], [1.0, 1.0, -0.0001], 1], [[0.0001, 0.0001, 0.0], [1.0, 0.0001, 0.0], 0], [[0.0001, 0.0001, -1.0], [1.0, 0.0001, -0.0001], 1], [[0.0, 0.0002, 0.0], [0.0, 1.0, 0.0], 0], [[0.0, 0.0002, -1.0], [0.0, 1.0, -0.0001], 1], [[0.0, 0.0001, 0.0], [0.0, 0.0001, 0.0], 0], [[0.0, 0.0001, -1.0], [0.0, 0.0001, -0.0001], 1], [[0.0, 0.0, 0.0001], [1.0, 0.0, 1.0], 0], [[0.0, -1.0, 0.0001], [1.0, -0.0001, 1.0], 1], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0], [[0.0, 0.0, -1.0], [1.0, 0.0, -0.0001], 1], [[0.0, -1.0, 0.0], [1.0, -0.0001, 0.0], 1], [[0.0, -1.0, -1.0], [1.0, -0.0001, -0.0001], 1], [[-1.0, 0.0001, 0.0001], [-0.0001, 1.0, 1.0], 1], [[-1.0, 0.0001, -1.0], [-0.0001, 1.0, 0.0], 1], [[-1.0, -1.0, 0.0001], [-0.0001, 0.0, 1.0], 1], [[-1.0, -1.0, -1.0], [-0.0001, 0.0, 0.0], 1]]
unit_sqr_mappings = []

inputDim = 3

def countNumUnitSqrSegments():
    num_unit_sqr_segments = 0
    for mapping in mappings:
        if (mapping[0][0] > mapping[1][0]) or (mapping[0][1] > mapping[1][1]) or (mapping[0][2] > mapping[1][2]):
            print(f"Lower bounds not less than upper bounds: {mapping}")
            exit(1)
        elif (mapping[0][0] >= 0 and mapping[0][1] >= 0 and mapping[0][2] >= 0) and (mapping[1][0] <= 1 and mapping[1][1] <= 1 and mapping[1][2] <= 1):
            num_unit_sqr_segments += 1
            unit_sqr_mappings.append(mapping)
    print(f"num_unit_sqr_segments: {num_unit_sqr_segments}")

def checkUnitSqrSegments():
    error = False
    for i in tqdm(range(0, 10000, 10)): # 10000 --> 0 to 1 with steps of EPSILON*10
        for j in range (0, 10000, 10):
            for k in range (0, 10000, 10):
                i = round(i / 10000, 4)
                j = round(j / 10000, 4)
                k = round(k / 10000, 4)
                # print(i,j,k)
                found = False
                for mapping in unit_sqr_mappings:
                    if (i >= mapping[0][0] and i <= mapping[1][0]) and (j >= mapping[0][1] and j <= mapping[1][1]) and (k >= mapping[0][2] and k <= mapping[1][2]):
                        found = True
                        if mapping[2] != 0:
                            print(f"Output should be 0: {mapping}")
                            exit(1)
                if not found:
                    print(f"No corresponding mapping for input: ({i}, {j}, {k})")
                    exit(1)
                        
countNumUnitSqrSegments()
checkUnitSqrSegments() 