# GLACIER AREAS
class GlacierBounds(object):
    """An object containing lon/lat pairs of vertices for polygons bounding
    separate glacier systems. The following areas are defined:

    EUROPE
    fen     : Mainlaind Fennoscandia
    eng     : England
    fullbar : Barents Sea including islands
    sval    : Svalbard (and adjacent Barents Sea)
    fjl     : Franz Josefland
    nz      : Novaya Zemlya
    bar     : Barents Sea excluding islands (marine areas)

    NORTH AMERICA
    laut    : Laurentian
    cor     : Cordilleran
    naf     : Southern fringe of Laurentian
    inu     : Inutian
    grn     : Greenland

    OTHER
    ice     : Iceland
    want    : West Antarctica
    eant    : East Antarctica
    """

    areaNames = ['eng', 'fen', 'fullbar', 'laur', 'cor', 'naf', 'inu', 'grn',
                    'ice', 'want', 'eant']

    # England
    eng = [(-5, 68.57), (-3.7, 68.57), (5.67, 49.82), (5.67, 49.82),
           (-10.23, 49.82), (-10.23, 60), (-5, 60)]

    # Mainland Fennoscandia
    #fen = [(-20, 50), (-15, 60), (-5, 70), (-5, 85), (180, 85), (180, 40),
    #       (-20, 40)]
    
    fen = [(5.67, 49.82), (-3.7, 68.57), (-0.78, 68.5), (10.47, 68.57), 
           (17.63, 70.29), (23.78, 71.18), (29.52, 70.9), (39.6, 68.11),
           (47.7, 63.87), (47.7, 40), (5.67, 40)]

    # Barents Sea including islands
    fullbar = [( -1.0, 81.33), (36, 85.07),   (66, 82.4),    (66, 78.),  
               (69.38, 76.01), (59.06, 72.11),(57.5, 69.06), (47.7, 63.87), 
               (39.6, 68.11), (29.52, 70.9), (23.78, 71.18), (17.63, 70.29), 
               (10.47, 68.57), (-0.78, 68.57)]

    # Svalbard
    sval = [(7.18, 81.33), (36, 81.33), (36, 76.64), (24, 75), (12.5, 75.7)]

    # Franz Josefland
    fjl = [(43, 79.9), (43, 82.4), (66, 82.4), (66, 78.0), (50, 78)]

    # Novaya Zemlya
    nz = [(50.93, 71.01), (50.93, 72.42), (58.12, 76.64), (66, 78),
          (69.38, 76.01), (59.06, 72.11), (57.5, 69.06)]

    # Barents Sea marine areas
    bar = [(-1.0, 81.33), (12.5, 75.7), (24, 75), (36, 76.64), (36, 85.07), 
           (43, 82.4),    (43, 79.9),   (50, 78), (66, 78.),   (58.12, 76.64), 
           (50.93, 72.42), (50.93, 71.01), (57.5, 69.06), (59.06, 72.11), 
           (69.38, 76.01), (66, 78), (80.7, 74.7), (75.8, 74.7), 
           (47.7, 63.87), (39.6, 68.11), (29.52, 70.9), (23.78, 71.18), 
           (17.63, 70.29), (10.47, 68.57), (-0.78, 68.57)]

    ###########  NORTH AMERICA  #########
    # Laurentian
    laur = [(-140, 74.5), (-70, 74.5), (-65, 70), (-57.5, 65), (-55, 60),
            (-50, 40), (-55, 60), (-73, 45), (-82.5, 45), (-90, 47.5),
            (-95, 48.5), (-105, 49.5), (-112.5, 52), (-125, 60), (-130, 65),
            (-140, 65)]

    # Cordilleran
    cor = [(-165, 65), (-130, 65), (-125, 60), (-120, 55), (-112.5, 47), 
           (-125, 45), (-165, 45)]

    # NAFringe
    naf = [(-112.5, 52), (-105, 49.5), (-95, 48.5), (-90, 47.5), (-82.5, 45),
           (-73, 45), (-55, 60), (-50, 54), (-50, 30), (-102.5, 30),
           (-102.5, 47), (-112.5, 47)]

    # Inutian
    inu = [(-140, 85), (-45, 85), (-55, 83), (-60, 82), (-65, 81), (-70, 79.5),
            (-75, 77), (-70, 74.5), (-140, 74.5)]

    # Greenland
    grn = [(-75, 77), (-70, 79.5), (-65, 81), (-60, 82), (-55, 83), (-45, 85),
           (-5, 85), (-5, 75), (-15, 70), (-30, 65), (-35, 60), (-55, 60), 
           (-57.5, 65), (-60, 70), (-70, 74.5), (-75, 77)]

    # Iceland
    ice = [(-35, 60), (-30, 65), (-15, 70), (-5, 75), (-5, 60), (-35, 60)]

    # West Antarctica
    want = [(-180, -55), (0, -55), (0, -90), (-180, -90)]

    # East Antarctica
    eant = [(-180, -84.5), (-150, -86), (-90, -87.5), (-7.5, -87.5), 
            (-7.5, -55), (172, -55), (172, -82.5), (180, -84.5), (180, -90),
            (-180, -90)]

    @classmethod
    def outputAsList(cls, names=None):
        names = names or cls.areaNames
        outputList = []
        for name in names:
            outputList.append({ 'name':name,
                                'vert':getattr(cls, name)})
        return outputList

