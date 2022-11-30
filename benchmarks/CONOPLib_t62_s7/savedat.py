def savedat(soln):
    outsoln = [[0] * 3 for _ in range(len(soln))]
    arrs = []
    for i in range(len(soln)):
        outsoln[i][0] = int(soln[i] / 2 + 1)
        outsoln[i][1] = soln[i] % 2 + 1
        outsoln[i][2] = i+1
    f = open('testsoln.dat', 'w')
    for i in range(len(soln)):
        jointsFrame = outsoln[i]  # 每行
        arrs.append(jointsFrame)
        for Ji in range(3):
            strNum = "        " + str(jointsFrame[Ji])
            f.write(strNum)
        f.write('\n')
    f.close()
