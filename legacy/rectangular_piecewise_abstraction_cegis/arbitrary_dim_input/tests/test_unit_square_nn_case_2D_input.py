from tqdm import tqdm

# Output of rectangular piecewise abstraction synthesizer for 2-dimensional input
# Schema: [ piecewise_segment_0, piecewise_segment_1 ] 
#   where a piecewise segment is [segment_lower_bound_coord, segment_upper_bound_coord, segment_class (0 or 1) ]
mappings = [[[0.0001, 1.0001], [100.0, 100.0], 1], [[1.0001, 1.0], [100.0, 1.0], 1], [[1.0001, 0.0001], [100.0, 0.9999], 1], [[0.0002, 1.0], [1.0, 1.0], 0], [[0.0002, 0.0001], [1.0, 0.9999], 0], [[1.0001, 0.0], [100.0, 0.0], 1], [[0.0003, 0.0], [1.0, 0.0], 0], [[0.0002, 0.0], [0.0002, 0.0], 0], [[0.0001, 0.0001], [0.0001, 1.0], 0], [[0.0001, 0.0], [0.0001, 0.0], 0], [[0.0, 1.0001], [0.0, 100.0], 1], [[0.0, 0.0001], [0.0, 1.0], 0], [[0.0, 0.0], [0.0, 0.0], 0], [[0.0, -100.0], [100.0, -0.0001], 1], [[-100.0, 0.0], [-0.0001, 100.0], 1], [[-100.0, -100.0], [-0.0001, -0.0001], 1]]

# Initialize an empty list to store mappings that fall within the unit square
unit_sqr_mappings = []

def countNumUnitSqrSegments():
    """Count and store mappings that fall within the unit square."""
    num_unit_sqr_segments = 0
    for mapping in mappings:
        # Check if mapping is invalid
        if (mapping[0][0] > mapping[1][0]) or (mapping[0][1] > mapping[1][1]):
            print(f"Lower bounds not less than upper bounds: {mapping}")
            exit(1)
        # Check if mapping is within unit square
        elif (mapping[0][0] >= 0 and mapping[0][1] >= 0) and (mapping[0][1] <= 1 and mapping[1][1] <= 1):
            num_unit_sqr_segments += 1
            unit_sqr_mappings.append(mapping)
    print(f"num_unit_sqr_segments: {num_unit_sqr_segments}")

def checkUnitSqrSegments():
    """Check if all points within the unit square have corresponding mappings."""
    for i in tqdm(range(0, 10000)):
        for j in range(0, 10000):
            i = round(i / 10000, 4)
            j = round(j / 10000, 4)

            found = False
            for mapping in unit_sqr_mappings:
                if (i >= mapping[0][0] and i <= mapping[1][0]) and (j >= mapping[0][1] and j <= mapping[1][1]):
                    found = True

                    # Check if the output class mapped to the segment is correct
                    if mapping[2] != 0:
                        print(f"Output should be 0: {mapping}")
                        exit(1)

            if not found:
                print(f"No corresponding mapping for input: ({i}, {j})")
                exit(1)
                    
countNumUnitSqrSegments()
checkUnitSqrSegments() 
print("\033[32mPASSED\033[0m")
