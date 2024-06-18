'''
beam sections:
(1) 350x200x9x16
(2) 350x200x9x20
(3) 350x200x12x22
(4) 375x200x12x25
(5) 375x250x12x25
(6) 375x250x12x28
(7) 400x250x16x32
(8) 400x250x16x36
(9) 400x300x16x36

column sections:
(1) 350x350x20
(2) 350x350x22
(3) 350x350x25
(4) 375x375x28
(5) 375x375x32
(6) 375x375x36
(7) 400x400x40
(8) 400x400x45
(9) 400x400x50
'''


# YIELDING_STRESS = 350 # 350 MPa = 350 x 10^3 kN/m^2, so My (kN x mm) = S (cm^3) x 350 (MPa) --> kN x mm
# convert Fy to kN, mm
YIELDING_STRESS = 350 * 1e+3 * 1e-6     # kN/mm^2
# so My then become --> My(kN x mm) = S (cm3) * 1e+3 * Fy (kN/mm^2) = kN x mm

# I-section shape factor = 1.12 ~ 1.14
# Rectangular shape factor = 1.5
BEAM_SHAPE_FACTOR = 1.12
COLUMN_SHAPE_FACTOR = 1.5

# I-beam effect shear area
# Asy = H * tw
# Asz = 5/6 * (2 * B * tf)
# Hollow rectangular shear area
# Asy = 2 * H * tw
# Asz = 2 * B * tf


beam_sections = []
column_sections = []


# Various Beam, Column section
beam_sections.append({
     'name': '350x200x9x16',
     'H(mm)': 350,
     'B(mm)': 200,
     't_f(mm)': 16,
     't_w(mm)': 9,
     'A(cm2)': 92.62,
     'J(cm4)': 64.333,
     'I_y(cm4)': 2135.265,
     'I_z(cm4)': 20274.4207,
     'S_y(cm3)': 213.526,
     'S_z(cm3)': 1158.538,
     'Av_y(cm2)': (350/10) * (9/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (16/10)),
     'My_z(kN-mm)': 1158.538 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1158.538 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 213.526 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1158.538 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [157, 227, 245]
})

beam_sections.append({
     'name': '350x200x9x20',
     'H(mm)': 350,
     'B(mm)': 200,
     't_f(mm)': 20,
     't_w(mm)': 9,
     'A(cm2)': 107.9,
     'J(cm4)': 116.386,
     'I_y(cm4)': 2668.549,
     'I_z(cm4)': 24040.991,
     'S_y(cm3)': 266.855,
     'S_z(cm3)': 1373.770,
     'Av_y(cm2)': (350/10) * (9/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (20/10)),
     'My_z(kN-mm)': 1373.770 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1373.770 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 266.855 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1373.770 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [133, 184, 245]
})

beam_sections.append({
     'name': '350x200x12x22',
     'H(mm)': 350,
     'B(mm)': 200,
     't_f(mm)': 22,
     't_w(mm)': 12,
     'A(cm2)': 123.72,
     'J(cm4)': 165.013,
     'I_y(cm4)': 2937.739,
     'I_z(cm4)': 26569.234,
     'S_y(cm3)': 293.773,
     'S_z(cm3)': 1518.241,
     'Av_y(cm2)': (350/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (22/10)),
     'My_z(kN-mm)': 1518.241 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1518.241 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 293.773 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1518.241 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [72, 92, 223]
})

beam_sections.append({
     'name': '375x200x12x25',
     'H(mm)': 375,
     'B(mm)': 200,
     't_f(mm)': 25,
     't_w(mm)': 12,
     'A(cm2)': 139.00,
     'J(cm4)': 237.133,
     'I_y(cm4)': 3338.013,
     'I_z(cm4)': 34109.895,
     'S_y(cm3)': 333.801,
     'S_z(cm3)': 1819.194,
     'Av_y(cm2)': (375/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (25/10)),
     'My_z(kN-mm)': 1819.194 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1819.194 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 333.801 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1819.194 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [37, 12, 211]
})

beam_sections.append({
     'name': '375x250x12x25',
     'H(mm)': 375,
     'B(mm)': 250,
     't_f(mm)': 25,
     't_w(mm)': 12,
     'A(cm2)': 164.00,
     'J(cm4)': 289.1216,
     'I_y(cm4)': 6515.096,
     'I_z(cm4)': 41779.166,
     'S_y(cm3)': 521.207,
     'S_z(cm3)': 2228.222,
     'Av_y(cm2)': (375/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (25/10)),
     'My_z(kN-mm)': 2228.222 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2228.222 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 521.207 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 2228.222 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [59, 6, 196]
})

beam_sections.append({
     'name': '375x250x12x28',
     'H(mm)': 375,
     'B(mm)': 250,
     't_f(mm)': 28,
     't_w(mm)': 12,
     'A(cm2)': 178.28,
     'J(cm4)': 394.666,
     'I_y(cm4)': 7296.260,
     'I_z(cm4)': 45480.792,
     'S_y(cm3)': 583.700,
     'S_z(cm3)': 2425.700,
     'Av_y(cm2)': (375/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (28/10)),
     'My_z(kN-mm)': 2425.700 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2425.700 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 583.700 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 2425.700 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [135, 25, 159]
})

beam_sections.append({
     'name': '400x250x12x32',
     'H(mm)': 400,
     'B(mm)': 250,
     't_f(mm)': 32,
     't_w(mm)': 12,
     'A(cm2)': 200.32,
     'J(cm4)': 580.693,
     'I_y(cm4)': 8338.171,
     'I_z(cm4)': 58099.438,
     'S_y(cm3)': 667.053,
     'S_z(cm3)': 2904.971,
     'Av_y(cm2)': (400/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (32/10)),
     'My_z(kN-mm)': 2904.971 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2904.971 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 667.053 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 2904.971 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [148, 28, 147]
})

beam_sections.append({
     'name': '400x250x16x36',
     'H(mm)': 400,
     'B(mm)': 250,
     't_f(mm)': 36,
     't_w(mm)': 16,
     'A(cm2)': 232.48,
     'J(cm4)': 859.520,
     'I_y(cm4)': 9386.195,
     'I_z(cm4)': 64522.606,
     'S_y(cm3)': 750.895,
     'S_z(cm3)': 3226.130,
     'Av_y(cm2)': (400/10) * (16/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (36/10)),
     'My_z(kN-mm)': 3226.130 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 3226.130 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 750.895 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 3226.130 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [111, 26, 109]
})

beam_sections.append({
     'name': '400x300x16x36',
     'H(mm)': 400,
     'B(mm)': 300,
     't_f(mm)': 36,
     't_w(mm)': 16,
     'A(cm2)': 268.48,
     'J(cm4)': 1015.040,
     'I_y(cm4)': 16211.195,
     'I_z(cm4)': 76486.126,
     'S_y(cm3)': 1080.746,
     'S_z(cm3)': 3824.306,
     'Av_y(cm2)': (400/10) * (16/10),
     'Av_z(cm2)': 5/6 * (2 * (300/10) * (36/10)),
     'My_z(kN-mm)': 3824.306 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 3824.306 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1080.746 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 3824.306 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [90, 24, 90]
})





column_sections.append({
     'name': '350x350x20',
     'H(mm)': 350,
     'B(mm)': 350,
     't_f(mm)': 20,
     't_w(mm)': 20,
     'A(cm2)': 264.00,
     'J(cm4)': 109744.000,
     'I_y(cm4)': 48092.000,
     'I_z(cm4)': 48092.000,
     'S_y(cm3)': 2748.114,
     'S_z(cm3)': 2748.114,
     'Av_y(cm2)': 2 * (350/10) * (20/10),
     'Av_z(cm2)': 2 * (350/10) * (20/10),
     'My_z(kN-mm)': 2748.114 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2748.114 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 2748.114 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [237, 245, 80]
})

column_sections.append({
     'name': '350x350x22',
     'H(mm)': 350,
     'B(mm)': 350,
     't_f(mm)': 22,
     't_w(mm)': 22,
     'A(cm2)': 288.64,
     'J(cm4)': 118822.334,
     'I_y(cm4)': 51987.912,
     'I_z(cm4)': 51987.912,
     'S_y(cm3)': 2970.737,
     'S_z(cm3)': 2970.737,
     'Av_y(cm2)': 2 * (350/10) * (22/10),
     'Av_z(cm2)': 2 * (350/10) * (22/10),
     'My_z(kN-mm)': 2970.737 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2970.737 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 2970.737 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [214, 230, 74]
})

column_sections.append({
     'name': '350x350x25',
     'H(mm)': 350,
     'B(mm)': 350,
     't_f(mm)': 25,
     't_w(mm)': 25,
     'A(cm2)': 325.00,
     'J(cm4)': 131835.937,
     'I_y(cm4)': 57552.083,
     'I_z(cm4)': 57552.083,
     'S_y(cm3)': 3288.690,
     'S_z(cm3)': 3288.690,
     'Av_y(cm2)': 2 * (350/10) * (25/10),
     'Av_z(cm2)': 2 * (350/10) * (25/10),
     'My_z(kN-mm)': 3288.690 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 3288.690 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 3288.690 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [191, 215, 68]
})

column_sections.append({
     'name': '375x375x28',
     'H(mm)': 375,
     'B(mm)': 375,
     't_f(mm)': 28,
     't_w(mm)': 28,
     'A(cm2)': 388.64,
     'J(cm4)': 294431.334,
     'I_y(cm4)': 78550.745,
     'I_z(cm4)': 78550.745,
     'S_y(cm3)': 4186.706,
     'S_z(cm3)': 4186.706,
     'Av_y(cm2)': 2 * (375/10) * (28/10),
     'Av_z(cm2)': 2 * (375/10) * (28/10),
     'My_z(kN-mm)': 4186.706 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 4186.706 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 4186.706 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [168, 199, 62]
})

column_sections.append({
     'name': '375x375x32',
     'H(mm)': 375,
     'B(mm)': 375,
     't_f(mm)': 32,
     't_w(mm)': 32,
     'A(cm2)': 439.039,
     'J(cm4)': 328010.243,
     'I_y(cm4)': 86836.989,
     'I_z(cm4)': 86836.989,
     'S_y(cm3)': 4631.306,
     'S_z(cm3)': 4631.306,
     'Av_y(cm2)': 2 * (375/10) * (32/10),
     'Av_z(cm2)': 2 * (375/10) * (32/10),
     'My_z(kN-mm)': 4631.306 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 4631.306 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 4631.306 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [146, 184, 56]
})

column_sections.append({
     'name': '375x375x36',
     'H(mm)': 375,
     'B(mm)': 375,
     't_f(mm)': 36,
     't_w(mm)': 36,
     'A(cm2)': 488.159,
     'J(cm4)': 359639.438,
     'I_y(cm4)': 94554.151,
     'I_z(cm4)': 94554.151,
     'S_y(cm3)': 5042.888,
     'S_z(cm3)': 5042.888,
     'Av_y(cm2)': 2 * (375/10) * (36/10),
     'Av_z(cm2)': 2 * (375/10) * (36/10),
     'My_z(kN-mm)': 5042.888 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 5042.888 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 5042.888 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [123, 169, 50]
})

column_sections.append({
     'name': '400x400x40',
     'H(mm)': 400,
     'B(mm)': 400,
     't_f(mm)': 40,
     't_w(mm)': 40,
     'A(cm2)': 576.00,
     'J(cm4)': 702464.000,
     'I_y(cm4)': 125951.999,
     'I_z(cm4)': 125951.999,
     'S_y(cm3)': 6297.599,
     'S_z(cm3)': 6297.599,
     'Av_y(cm2)': 2 * (400/10) * (40/10),
     'Av_z(cm2)': 2 * (400/10) * (40/10),
     'My_z(kN-mm)': 6297.599 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 6297.599 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 6297.599 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [100, 154, 44]
})

column_sections.append({
     'name': '400x400x45',
     'H(mm)': 400,
     'B(mm)': 400,
     't_f(mm)': 45,
     't_w(mm)': 45,
     'A(cm2)': 639.00,
     'J(cm4)': 769292.437,
     'I_y(cm4)': 136373.250,
     'I_z(cm4)': 136373.250,
     'S_y(cm3)': 6818.662,
     'S_z(cm3)': 6818.662,
     'Av_y(cm2)': 2 * (400/10) * (45/10),
     'Av_z(cm2)': 2 * (400/10) * (45/10),
     'My_z(kN-mm)': 6818.662 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 6818.662 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 6818.662 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [77, 138, 38]
})

column_sections.append({
     'name': '400x400x50',
     'H(mm)': 400,
     'B(mm)': 400,
     't_f(mm)': 50,
     't_w(mm)': 50,
     'A(cm2)': 700.00,
     'J(cm4)': 831865.000,
     'I_y(cm4)': 145833.333,
     'I_z(cm4)': 145833.333,
     'S_y(cm3)': 7291.666,
     'S_z(cm3)': 7291.666,
     'Av_y(cm2)': 2 * (400/10) * (50/10),
     'Av_z(cm2)': 2 * (400/10) * (50/10),
     'My_z(kN-mm)': 7291.666 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 7291.666 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 7291.666 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [54, 123, 32]
})







"""
-----------------------------------------------------------------------------------------------------------------
"""





beam_section_dict = {}
column_section_dict = {}




beam_section_dict['350x200x9x16'] = {
     'H(mm)': 350,
     'B(mm)': 200,
     't_f(mm)': 16,
     't_w(mm)': 9,
     'A(cm2)': 92.62,
     'J(cm4)': 64.333,
     'I_y(cm4)': 2135.265,
     'I_z(cm4)': 20274.4207,
     'S_y(cm3)': 213.526,
     'S_z(cm3)': 1158.538,
     'Av_y(cm2)': (350/10) * (9/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (16/10)),
     'My_z(kN-mm)': 1158.538 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1158.538 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 213.526 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1158.538 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [157, 227, 245]
}

beam_section_dict['350x200x9x20'] = {
     'H(mm)': 350,
     'B(mm)': 200,
     't_f(mm)': 20,
     't_w(mm)': 9,
     'A(cm2)': 107.9,
     'J(cm4)': 116.386,
     'I_y(cm4)': 2668.549,
     'I_z(cm4)': 24040.991,
     'S_y(cm3)': 266.855,
     'S_z(cm3)': 1373.770,
     'Av_y(cm2)': (350/10) * (9/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (20/10)),
     'My_z(kN-mm)': 1373.770 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1373.770 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 266.855 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1373.770 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [133, 184, 245]
}

beam_section_dict['350x200x12x22'] = {
     'H(mm)': 350,
     'B(mm)': 200,
     't_f(mm)': 22,
     't_w(mm)': 12,
     'A(cm2)': 123.72,
     'J(cm4)': 165.013,
     'I_y(cm4)': 2937.739,
     'I_z(cm4)': 26569.234,
     'S_y(cm3)': 293.773,
     'S_z(cm3)': 1518.241,
     'Av_y(cm2)': (350/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (22/10)),
     'My_z(kN-mm)': 1518.241 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1518.241 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 293.773 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1518.241 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [72, 92, 223]
}

beam_section_dict['375x200x12x25'] = {
     'H(mm)': 375,
     'B(mm)': 200,
     't_f(mm)': 25,
     't_w(mm)': 12,
     'A(cm2)': 139.00,
     'J(cm4)': 237.133,
     'I_y(cm4)': 3338.013,
     'I_z(cm4)': 34109.895,
     'S_y(cm3)': 333.801,
     'S_z(cm3)': 1819.194,
     'Av_y(cm2)': (375/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (25/10)),
     'My_z(kN-mm)': 1819.194 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1819.194 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 333.801 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1819.194 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [37, 12, 211]
}

beam_section_dict['375x250x12x25'] = {
     'H(mm)': 375,
     'B(mm)': 250,
     't_f(mm)': 25,
     't_w(mm)': 12,
     'A(cm2)': 164.00,
     'J(cm4)': 289.1216,
     'I_y(cm4)': 6515.096,
     'I_z(cm4)': 41779.166,
     'S_y(cm3)': 521.207,
     'S_z(cm3)': 2228.222,
     'Av_y(cm2)': (375/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (25/10)),
     'My_z(kN-mm)': 2228.222 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2228.222 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 521.207 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 2228.222 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [59, 6, 196]
}

beam_section_dict['375x250x12x28'] = {
     'H(mm)': 375,
     'B(mm)': 250,
     't_f(mm)': 28,
     't_w(mm)': 12,
     'A(cm2)': 178.28,
     'J(cm4)': 394.666,
     'I_y(cm4)': 7296.260,
     'I_z(cm4)': 45480.792,
     'S_y(cm3)': 583.700,
     'S_z(cm3)': 2425.700,
     'Av_y(cm2)': (375/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (28/10)),
     'My_z(kN-mm)': 2425.700 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2425.700 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 583.700 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 2425.700 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [135, 25, 159]
}

beam_section_dict['400x250x12x32'] = {
     'H(mm)': 400,
     'B(mm)': 250,
     't_f(mm)': 32,
     't_w(mm)': 12,
     'A(cm2)': 200.32,
     'J(cm4)': 580.693,
     'I_y(cm4)': 8338.171,
     'I_z(cm4)': 58099.438,
     'S_y(cm3)': 667.053,
     'S_z(cm3)': 2904.971,
     'Av_y(cm2)': (400/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (32/10)),
     'My_z(kN-mm)': 2904.971 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2904.971 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 667.053 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 2904.971 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [148, 28, 147]
}

beam_section_dict['400x250x16x36'] = {
     'H(mm)': 400,
     'B(mm)': 250,
     't_f(mm)': 36,
     't_w(mm)': 16,
     'A(cm2)': 232.48,
     'J(cm4)': 859.520,
     'I_y(cm4)': 9386.195,
     'I_z(cm4)': 64522.606,
     'S_y(cm3)': 750.895,
     'S_z(cm3)': 3226.130,
     'Av_y(cm2)': (400/10) * (16/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (36/10)),
     'My_z(kN-mm)': 3226.130 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 3226.130 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 750.895 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 3226.130 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [111, 26, 109]
}

beam_section_dict['400x300x16x36'] = {
     'H(mm)': 400,
     'B(mm)': 300,
     't_f(mm)': 36,
     't_w(mm)': 16,
     'A(cm2)': 268.48,
     'J(cm4)': 1015.040,
     'I_y(cm4)': 16211.195,
     'I_z(cm4)': 76486.126,
     'S_y(cm3)': 1080.746,
     'S_z(cm3)': 3824.306,
     'Av_y(cm2)': (400/10) * (16/10),
     'Av_z(cm2)': 5/6 * (2 * (300/10) * (36/10)),
     'My_z(kN-mm)': 3824.306 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 3824.306 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1080.746 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 3824.306 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [90, 24, 90]
}





column_section_dict['350x350x20'] = {
     'H(mm)': 350,
     'B(mm)': 350,
     't_f(mm)': 20,
     't_w(mm)': 20,
     'A(cm2)': 264.00,
     'J(cm4)': 109744.000,
     'I_y(cm4)': 48092.000,
     'I_z(cm4)': 48092.000,
     'S_y(cm3)': 2748.114,
     'S_z(cm3)': 2748.114,
     'Av_y(cm2)': 2 * (350/10) * (20/10),
     'Av_z(cm2)': 2 * (350/10) * (20/10),
     'My_z(kN-mm)': 2748.114 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2748.114 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 2748.114 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [237, 245, 80]
}

column_section_dict['350x350x22'] = {
     'H(mm)': 350,
     'B(mm)': 350,
     't_f(mm)': 22,
     't_w(mm)': 22,
     'A(cm2)': 288.64,
     'J(cm4)': 118822.334,
     'I_y(cm4)': 51987.912,
     'I_z(cm4)': 51987.912,
     'S_y(cm3)': 2970.737,
     'S_z(cm3)': 2970.737,
     'Av_y(cm2)': 2 * (350/10) * (22/10),
     'Av_z(cm2)': 2 * (350/10) * (22/10),
     'My_z(kN-mm)': 2970.737 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2970.737 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 2970.737 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [214, 230, 74]
}

column_section_dict['350x350x25'] = {
     'H(mm)': 350,
     'B(mm)': 350,
     't_f(mm)': 25,
     't_w(mm)': 25,
     'A(cm2)': 325.00,
     'J(cm4)': 131835.937,
     'I_y(cm4)': 57552.083,
     'I_z(cm4)': 57552.083,
     'S_y(cm3)': 3288.690,
     'S_z(cm3)': 3288.690,
     'Av_y(cm2)': 2 * (350/10) * (25/10),
     'Av_z(cm2)': 2 * (350/10) * (25/10),
     'My_z(kN-mm)': 3288.690 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 3288.690 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 3288.690 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [191, 215, 68]
}

column_section_dict['375x375x28'] = {
     'H(mm)': 375,
     'B(mm)': 375,
     't_f(mm)': 28,
     't_w(mm)': 28,
     'A(cm2)': 388.64,
     'J(cm4)': 294431.334,
     'I_y(cm4)': 78550.745,
     'I_z(cm4)': 78550.745,
     'S_y(cm3)': 4186.706,
     'S_z(cm3)': 4186.706,
     'Av_y(cm2)': 2 * (375/10) * (28/10),
     'Av_z(cm2)': 2 * (375/10) * (28/10),
     'My_z(kN-mm)': 4186.706 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 4186.706 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 4186.706 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [168, 199, 62]
}

column_section_dict['375x375x32'] = {
     'H(mm)': 375,
     'B(mm)': 375,
     't_f(mm)': 32,
     't_w(mm)': 32,
     'A(cm2)': 439.039,
     'J(cm4)': 328010.243,
     'I_y(cm4)': 86836.989,
     'I_z(cm4)': 86836.989,
     'S_y(cm3)': 4631.306,
     'S_z(cm3)': 4631.306,
     'Av_y(cm2)': 2 * (375/10) * (32/10),
     'Av_z(cm2)': 2 * (375/10) * (32/10),
     'My_z(kN-mm)': 4631.306 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 4631.306 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 4631.306 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [146, 184, 56]
}

column_section_dict['375x375x36'] = {
     'H(mm)': 375,
     'B(mm)': 375,
     't_f(mm)': 36,
     't_w(mm)': 36,
     'A(cm2)': 488.159,
     'J(cm4)': 359639.438,
     'I_y(cm4)': 94554.151,
     'I_z(cm4)': 94554.151,
     'S_y(cm3)': 5042.888,
     'S_z(cm3)': 5042.888,
     'Av_y(cm2)': 2 * (375/10) * (36/10),
     'Av_z(cm2)': 2 * (375/10) * (36/10),
     'My_z(kN-mm)': 5042.888 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 5042.888 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 5042.888 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [123, 169, 50]
}

column_section_dict['400x400x40'] = {
     'H(mm)': 400,
     'B(mm)': 400,
     't_f(mm)': 40,
     't_w(mm)': 40,
     'A(cm2)': 576.00,
     'J(cm4)': 702464.000,
     'I_y(cm4)': 125951.999,
     'I_z(cm4)': 125951.999,
     'S_y(cm3)': 6297.599,
     'S_z(cm3)': 6297.599,
     'Av_y(cm2)': 2 * (400/10) * (40/10),
     'Av_z(cm2)': 2 * (400/10) * (40/10),
     'My_z(kN-mm)': 6297.599 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 6297.599 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 6297.599 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [100, 154, 44]
}

column_section_dict['400x400x45'] = {
     'H(mm)': 400,
     'B(mm)': 400,
     't_f(mm)': 45,
     't_w(mm)': 45,
     'A(cm2)': 639.00,
     'J(cm4)': 769292.437,
     'I_y(cm4)': 136373.250,
     'I_z(cm4)': 136373.250,
     'S_y(cm3)': 6818.662,
     'S_z(cm3)': 6818.662,
     'Av_y(cm2)': 2 * (400/10) * (45/10),
     'Av_z(cm2)': 2 * (400/10) * (45/10),
     'My_z(kN-mm)': 6818.662 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 6818.662 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 6818.662 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [77, 138, 38]
}

column_section_dict['400x400x50'] = {
     'H(mm)': 400,
     'B(mm)': 400,
     't_f(mm)': 50,
     't_w(mm)': 50,
     'A(cm2)': 700.00,
     'J(cm4)': 831865.000,
     'I_y(cm4)': 145833.333,
     'I_z(cm4)': 145833.333,
     'S_y(cm3)': 7291.666,
     'S_z(cm3)': 7291.666,
     'Av_y(cm2)': 2 * (400/10) * (50/10),
     'Av_z(cm2)': 2 * (400/10) * (50/10),
     'My_z(kN-mm)': 7291.666 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 7291.666 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 7291.666 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [54, 123, 32]
}