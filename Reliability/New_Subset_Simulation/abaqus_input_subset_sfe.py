# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__
import numpy as np
import math


#
# def polevl(x, coefs, N):
#     ans = 0
#     power = len(coefs) - 1
#     for coef in coefs:
#         ans += coef * x ** power
#         power -= 1
#     return ans
#
#
# def p1evl(x, coefs, N):
#     return polevl(x, [1] + coefs, N)
#
# # From scipy special/cephes/ndrti.c
# def ndtri(y):
#     # approximation for 0 <= abs(z - 0.5) <= 3/8
#     P0 = [
#         -5.99633501014107895267E1,
#         9.80010754185999661536E1,
#         -5.66762857469070293439E1,
#         1.39312609387279679503E1,
#         -1.23916583867381258016E0,
#     ]
#
#     Q0 = [
#         1.95448858338141759834E0,
#         4.67627912898881538453E0,
#         8.63602421390890590575E1,
#         -2.25462687854119370527E2,
#         2.00260212380060660359E2,
#         -8.20372256168333339912E1,
#         1.59056225126211695515E1,
#         -1.18331621121330003142E0,
#     ]
#
#     # Approximation for interval z = sqrt(-2 log y ) between 2 and 8
#     # i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
#     P1 = [
#         4.05544892305962419923E0,
#         3.15251094599893866154E1,
#         5.71628192246421288162E1,
#         4.40805073893200834700E1,
#         1.46849561928858024014E1,
#         2.18663306850790267539E0,
#         -1.40256079171354495875E-1,
#         -3.50424626827848203418E-2,
#         -8.57456785154685413611E-4,
#     ]
#
#     Q1 = [
#         1.57799883256466749731E1,
#         4.53907635128879210584E1,
#         4.13172038254672030440E1,
#         1.50425385692907503408E1,
#         2.50464946208309415979E0,
#         -1.42182922854787788574E-1,
#         -3.80806407691578277194E-2,
#         -9.33259480895457427372E-4,
#     ]
#
#     # Approximation for interval z = sqrt(-2 log y ) between 8 and 64
#     # i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
#     P2 = [
#         3.23774891776946035970E0,
#         6.91522889068984211695E0,
#         3.93881025292474443415E0,
#         1.33303460815807542389E0,
#         2.01485389549179081538E-1,
#         1.23716634817820021358E-2,
#         3.01581553508235416007E-4,
#         2.65806974686737550832E-6,
#         6.23974539184983293730E-9,
#     ]
#
#     Q2 = [
#         6.02427039364742014255E0,
#         3.67983563856160859403E0,
#         1.37702099489081330271E0,
#         2.16236993594496635890E-1,
#         1.34204006088543189037E-2,
#         3.28014464682127739104E-4,
#         2.89247864745380683936E-6,
#         6.79019408009981274425E-9,
#     ]
#
#     s2pi = 2.50662827463100050242
#     code = 1
#
#     if y > (1.0 - 0.13533528323661269189):  # 0.135... = exp(-2)
#         y = 1.0 - y
#         code = 0
#
#     if y > 0.13533528323661269189:
#         y = y - 0.5
#         y2 = y * y
#         x = y + y * (y2 * polevl(y2, P0, 4) / p1evl(y2, Q0, 8))
#         x = x * s2pi
#         return x
#
#     x = math.sqrt(-2.0 * math.log(y))
#     x0 = x - math.log(x) / x
#
#     z = 1.0 / x
#     if x < 8.0:  # y > exp(-32) = 1.2664165549e-14
#         x1 = z * polevl(z, P1, 8) / p1evl(z, Q1, 8)
#     else:
#         x1 = z * polevl(z, P2, 8) / p1evl(z, Q2, 8)
#
#     x = x0 - x1
#     if code != 0:
#         x = -x
#
#     return x
#
# def inv_erf(z):
#     if z < -1 or z > 1:
#         raise ValueError("`z` must be between -1 and 1 inclusive")
#
#     if z == 0:
#         return 0
#     if z == 1:
#         return math.inf
#     if z == -1:
#         return -math.inf
#
#     result = ndtri((z + 1) / 2.0) / math.sqrt(2)
#
#     return result

def phi(x=0):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def time_temperature_curve(qtd=None):
    # Transform from N(0, 1) to Unif(50,400)
    qtd = 50 + 400*phi(qtd)
    print('Fire load density: ', qtd)

    # Define the other parameters of the curve
    O = 0.14
    b = 1500
    typ = 'medium'
    # Heating phase
    t_lim = 20 / 60.0
    if typ.lower() == 'slow':
        t_lim = 25 / 60.0
    elif typ.lower() == 'medium':
        t_lim = 20 / 60.0
    elif typ.lower() == 'fast':
        t_lim = 15 / 60.0
    gamma = ((O / b) ** 2) / ((0.04 / 1160) ** 2)
    t_max = max((0.2e-3 * qtd / O, t_lim))
    if t_max == t_lim:
        O_lim = 0.1e-3 * qtd / t_lim
        gamma_lim = ((O_lim / b) ** 2) / ((0.04 / 1160) ** 2)
        gamma = gamma_lim
    t_max_star = t_max * gamma
    n_points = 50
    t = np.linspace(0, t_max_star, n_points)
    t_star = t
    theta_room = 0
    theta_g = theta_room + 1325 * (1 - 0.324 * np.exp(-0.2 * t_star) -
                                   0.204 * np.exp(-1.7 * t_star) - 0.472 * np.exp(-19 * t_star))
    # Cooling phase
    theta_max = max(theta_g)
    t_max_star = 0.2e-3 * qtd / O * gamma
    # TODO: Check the cooling phase when qtd = 125
    x = 1.0
    if t_max == t_lim:
        x = t_lim * gamma / t_max_star
    if t_max_star <= 0.5:
        t_star_room = (theta_max - theta_room) / 625 + t_max_star * x
        # t_star_cooling = np.linspace(t_max_star, t_star_room, n_points + 1)
        # theta_cooling = theta_max - 625 * (t_star_cooling - t_max_star * x)
    elif t_max_star < 2.0:
        t_star_room = (theta_max - theta_room) / (250 * (3 - t_max_star)) + t_max_star * x
        # t_star_cooling = np.linspace(t_max_star, t_star_room, n_points + 1)
        # theta_cooling = theta_max - 250 * (3 - t_max_star) * (t_star_cooling - t_max_star * x)
    elif t_max_star >= 2.0:
        t_star_room = (theta_max - theta_room) / 250 + t_max_star * x
        # t_star_cooling = np.linspace(t_max_star, t_star_room, n_points + 1)
        # theta_cooling = theta_max - 250 * (t_star_cooling - t_max_star * x)
        # theta_g = np.append(theta_g, theta_cooling)
    theta_g = np.append(theta_g, theta_room)
    t_star = np.append(t_star, t_star_room)
    t = t_star / gamma * 60.0
    t, idx = np.unique(t, return_index=True)
    theta_g = theta_g[idx]
    max_time = max(t)
    t /= max_time
    time_temp_curve = []
    for i in range(len(t)):
        time_temp_curve.append((t[i], theta_g[i],))
    time_temp_curve = tuple(time_temp_curve)
    return time_temp_curve, max_time


def FireProblem():
    # Import Abaqus modules
    import section
    import regionToolset
    import displayGroupMdbToolset as dgm
    import part
    import material
    import assembly
    import step
    import interaction
    import load
    import mesh
    import optimization
    import job
    import sketch
    import visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    import connectorBehavior

    # Create new model
    mdb.Model(name='FireBenchmark', modelType=STANDARD_EXPLICIT)
    session.viewports['Viewport: 1'].setValues(displayedObject=None)

    # Create part
    s = mdb.models['FireBenchmark'].ConstrainedSketch(name='__profile__',
                                                      sheetSize=5.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.rectangle(point1=(0.0, 0.0), point2=(1.0, 0.035))
    p = mdb.models['FireBenchmark'].Part(name='Beam', dimensionality=THREE_D,
                                         type=DEFORMABLE_BODY)
    p = mdb.models['FireBenchmark'].parts['Beam']
    p.BaseShell(sketch=s)
    s.unsetPrimaryObject()
    p = mdb.models['FireBenchmark'].parts['Beam']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['FireBenchmark'].sketches['__profile__']
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON,
                                                           engineeringFeatures=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)

    # Create material
    mdb.models['FireBenchmark'].Material(name='Steel')
    mdb.models['FireBenchmark'].materials['Steel'].Elastic(
        temperatureDependency=ON, table=((207000000000.0, 0.33, 0.0), (1.0,
                                                                       0.33, 1200.0)))
    yield_strength = <fy>*0.07*250e6 + 250e6
    print('Yield strength: ', yield_strength)
    mdb.models['FireBenchmark'].materials['Steel'].Plastic(
        temperatureDependency=ON, table=((yield_strength, 0.0, 0.0), (1.0, 0.0,
                                                              1200.0)))
    mdb.models['FireBenchmark'].materials['Steel'].Expansion(table=((1.2e-05,
                                                                     ),))

    # Create section
    mdb.models['FireBenchmark'].HomogeneousShellSection(name='ShellSection',
                                                        preIntegrate=OFF, material='Steel', thicknessType=UNIFORM,
                                                        thickness=0.035, thicknessField='',
                                                        idealization=NO_IDEALIZATION,
                                                        poissonDefinition=DEFAULT, thicknessModulus=None,
                                                        temperature=GRADIENT,
                                                        useDensity=OFF, integrationRule=SIMPSON, numIntPts=5)

    # Assign section
    p = mdb.models['FireBenchmark'].parts['Beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#1 ]',), )
    region = p.Set(faces=faces, name='BeamWhole')
    p = mdb.models['FireBenchmark'].parts['Beam']
    p.SectionAssignment(region=region, sectionName='ShellSection', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)

    # Assembly
    a = mdb.models['FireBenchmark'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    a = mdb.models['FireBenchmark'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['FireBenchmark'].parts['Beam']

    # Create instance of part in assembly
    a.Instance(name='Beam-1', part=p, dependent=OFF)
    a = mdb.models['FireBenchmark'].rootAssembly
    a.translate(instanceList=('Beam-1',), vector=(0.0, -0.0175, 0.0))
    a = mdb.models['FireBenchmark'].rootAssembly

    # Create partition to define midpoint
    f1 = a.instances['Beam-1'].faces
    pickedFaces = f1.getSequenceFromMask(mask=('[#1 ]',), )
    e1 = a.instances['Beam-1'].edges
    a.PartitionFaceByShortestPath(faces=pickedFaces,
                                  point1=a.instances['Beam-1'].InterestingPoint(edge=e1[2], rule=MIDDLE),
                                  point2=a.instances['Beam-1'].InterestingPoint(edge=e1[0], rule=MIDDLE))
    a = mdb.models['FireBenchmark'].rootAssembly
    v1 = a.instances['Beam-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#2 ]',), )
    a.Set(vertices=verts1, name='Midpoint')

    # Create reference points
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON,
                                                               constraints=ON, connectors=ON, engineeringFeatures=ON)
    a = mdb.models['FireBenchmark'].rootAssembly
    e11 = a.instances['Beam-1'].edges
    a.ReferencePoint(point=a.instances['Beam-1'].InterestingPoint(edge=e11[5],
                                                                  rule=MIDDLE))
    a = mdb.models['FireBenchmark'].rootAssembly
    e1 = a.instances['Beam-1'].edges
    a.ReferencePoint(point=a.instances['Beam-1'].InterestingPoint(edge=e1[2],
                                                                  rule=MIDDLE))
    mdb.models['FireBenchmark'].rootAssembly.features.changeKey(fromName='RP-1',
                                                                toName='RP_LHS')
    mdb.models['FireBenchmark'].rootAssembly.features.changeKey(fromName='RP-2',
                                                                toName='RP_RHS')

    # Create constraints
    a = mdb.models['FireBenchmark'].rootAssembly
    r1 = a.referencePoints
    refPoints1 = (r1[6],)
    region1 = a.Set(referencePoints=refPoints1, name='RP_LHS')
    a = mdb.models['FireBenchmark'].rootAssembly
    e1 = a.instances['Beam-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#20 ]',), )
    region2 = a.Set(edges=edges1, name='LHS')
    mdb.models['FireBenchmark'].Coupling(name='LHS', controlPoint=region1,
                                         surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                                         localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
    a = mdb.models['FireBenchmark'].rootAssembly
    r1 = a.referencePoints
    refPoints1 = (r1[7],)
    region1 = a.Set(referencePoints=refPoints1, name='RP_RHS')
    a = mdb.models['FireBenchmark'].rootAssembly
    e1 = a.instances['Beam-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#4 ]',), )
    region2 = a.Set(edges=edges1, name='RHS')
    mdb.models['FireBenchmark'].Coupling(name='RHS', controlPoint=region1,
                                         surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                                         localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

    # # Create axial spring
    # a = mdb.models['FireBenchmark'].rootAssembly
    # region = a.sets['RP_RHS']
    # mdb.models['FireBenchmark'].rootAssembly.engineeringFeatures.SpringDashpotToGround(
    #     name='AxialSpring', region=region, orientation=None, dof=1,
    #     springBehavior=ON, springStiffness=0.1901812500, dashpotBehavior=OFF,
    #     dashpotCoefficient=0.0)
    # session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=OFF,
    #                                                            constraints=OFF, connectors=OFF, engineeringFeatures=OFF,
    #                                                            adaptiveMeshConstraints=ON)

    # Create steps
    mdb.models['FireBenchmark'].StaticStep(name='MechStep', previous='Initial',
                                           initialInc=0.1, maxInc=0.1, nlgeom=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='MechStep')
    mdb.models['FireBenchmark'].StaticStep(name='ThermalStep', previous='MechStep',
                                           maxNumInc=1000, initialInc=0.0001, maxInc=0.02)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='ThermalStep')

    # Create field output request
    mdb.models['FireBenchmark'].fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U'))

    # Create history output requests
    mdb.models['FireBenchmark'].historyOutputRequests['H-Output-1'].setValues(
        variables=('ALLAE', 'ALLIE', 'ALLKE', 'ALLPD', 'ALLSE', 'ALLWK',
                   'ETOTAL'))
    regionDef = mdb.models['FireBenchmark'].rootAssembly.sets['Midpoint']
    mdb.models['FireBenchmark'].HistoryOutputRequest(name='Midpoint',
                                                     createStepName='ThermalStep', variables=('U2', 'NT'),
                                                     region=regionDef,
                                                     sectionPoints=DEFAULT, rebar=EXCLUDE)
    regionDef = mdb.models['FireBenchmark'].rootAssembly.sets['RP_LHS']
    mdb.models['FireBenchmark'].HistoryOutputRequest(name='RP_LHS',
                                                     createStepName='ThermalStep', variables=('RF1',), region=regionDef,
                                                     sectionPoints=DEFAULT, rebar=EXCLUDE)

    # Create amplitude of the time-temperature curve
    time_temp_curve, max_time = time_temperature_curve(qtd=<qtd>)
    mdb.models['FireBenchmark'].TabularAmplitude(name='TimeTempCurve',
                                                 timeSpan=STEP, smooth=SOLVER_DEFAULT, data=time_temp_curve)

    # Apply mechanical load
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, predefinedFields=ON, connectors=ON,
                                                               adaptiveMeshConstraints=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='MechStep')
    a = mdb.models['FireBenchmark'].rootAssembly
    s1 = a.instances['Beam-1'].faces
    side1Faces1 = s1.getSequenceFromMask(mask=('[#3 ]',), )
    region = a.Surface(side1Faces=side1Faces1, name='BeamWeb')
    mdb.models['FireBenchmark'].SurfaceTraction(name='Traction',
                                                createStepName='MechStep', region=region, magnitude=94444.44,
                                                directionVector=((0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
                                                distributionType=UNIFORM, field='', localCsys=None, traction=GENERAL,
                                                follower=OFF, resultant=ON)

    # Apply displacement boundary condition to reference point at LHS
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
    a = mdb.models['FireBenchmark'].rootAssembly
    region = a.sets['RP_LHS']
    mdb.models['FireBenchmark'].DisplacementBC(name='LHS',
                                               createStepName='Initial', region=region, u1=SET, u2=SET, u3=SET,
                                               ur1=SET, ur2=SET, ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM,
                                               fieldName='', localCsys=None)

    # Apply displacement boundary condition to reference point at RHS
    a = mdb.models['FireBenchmark'].rootAssembly
    region = a.sets['RP_RHS']
    mdb.models['FireBenchmark'].DisplacementBC(name='RP_RHS',
                                               createStepName='Initial', region=region, u1=UNSET, u2=SET, u3=SET,
                                               ur1=SET, ur2=SET, ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM,
                                               fieldName='', localCsys=None)

    # Apply temperature field during the thermal step
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='ThermalStep')
    a = mdb.models['FireBenchmark'].rootAssembly
    region = a.instances['Beam-1'].sets['BeamWhole']
    mdb.models['FireBenchmark'].Temperature(name='TempField',
                                            createStepName='ThermalStep', region=region, distributionType=UNIFORM,
                                            crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1.0,),
                                            amplitude='TimeTempCurve')

    # Define edge seeds for mesh
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF,
                                                               bcs=OFF, predefinedFields=OFF, connectors=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(meshTechnique=ON)
    a = mdb.models['FireBenchmark'].rootAssembly
    e1 = a.instances['Beam-1'].edges
    pickedEdges = e1.getSequenceFromMask(mask=('[#25 ]',), )
    a.seedEdgeByNumber(edges=pickedEdges, number=6, constraint=FINER)
    a = mdb.models['FireBenchmark'].rootAssembly
    e1 = a.instances['Beam-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#25 ]',), )
    a.Set(edges=edges1, name='VerticalEdges')
    a = mdb.models['FireBenchmark'].rootAssembly
    e1 = a.instances['Beam-1'].edges
    pickedEdges = e1.getSequenceFromMask(mask=('[#5a ]',), )
    a.seedEdgeBySize(edges=pickedEdges, size=0.02, deviationFactor=0.1,
                     constraint=FINER)
    a = mdb.models['FireBenchmark'].rootAssembly
    e1 = a.instances['Beam-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#5a ]',), )
    a.Set(edges=edges1, name='HorizontalEdges')

    # Assign element type
    elemType1 = mesh.ElemType(elemCode=S4, elemLibrary=STANDARD,
                              secondOrderAccuracy=OFF)
    elemType2 = mesh.ElemType(elemCode=S3, elemLibrary=STANDARD)
    a = mdb.models['FireBenchmark'].rootAssembly
    f1 = a.instances['Beam-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#3 ]',), )
    pickedRegions = (faces1,)
    a.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))

    # Generate mesh
    a = mdb.models['FireBenchmark'].rootAssembly
    partInstances = (a.instances['Beam-1'],)
    a.generateMesh(regions=partInstances)

    # Create the job
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    jobname = 'fire_analysis'
    mdb.Job(name=jobname, model='FireBenchmark', description='',
            type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None,
            memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
            modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
            scratch='', multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)

    # Submit the job
    mdb.jobs[jobname].submit(consistencyChecking=OFF)


if __name__ == "__main__":
    FireProblem()
