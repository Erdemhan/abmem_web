import math

e = math.e

def calculate(holy, mcp24, mcp168, mcp672, der, tg, ng, lig, rhyd, icoal, sol, asph, bcoal, imex, ngp, ist):
    s1 = (0.518*mcp24) + (0.1224*mcp168) - (0.001101*rhyd) - (0.05642*ngp) - (0.3794*e)**(rhyd/icoal)
    s2 = ( (2.755*e) + (15*mcp24) + (2.755*e) + (15*mcp168) ) / ( (4.398*e) + (12*ng) + (8.796*e) + (12*rhyd) - (4.398*e) + (12*tg) + (12*holy*sol*asph)  )
    s3 = (0.0005505*holy*ngp) + (0.002202*der*ngp) + (1.461*math.sqrt(mcp24+ist+(mcp672/der)))
    s4 = ((0.3794*mcp24)/der) + ((1.004*mcp168)/der) + (0.5018*math.sqrt(mcp24+ngp)) + (0.0005505*(mcp672-7.777)**2) + (2.067*math.sqrt(ngp))
    s5 = ((7.734*e) + (15*holy*(mcp168+mcp672+der))) / ((3.518*e) + (13*ng) + (7.037*e) + (13*ist) + (3.518*e) + (13*abs(mcp168-imex)))
    s6p1 = ((2.426*e) + (15*holy*bcoal*(mcp24-sol))) / ((9.007*e) + (15*ng) + (1.801*e) + (16*mcp24) - (9.007*e) + (15*bcoal))
    s6p2 = ((0.03238*(mcp24**2)*lig*sol) / (der*(mcp168+ng)*(mcp168+mcp672+sol))) - 24.21

    return s1 - s2 - s3 - s4 + s5 - s6p1 - s6p2
