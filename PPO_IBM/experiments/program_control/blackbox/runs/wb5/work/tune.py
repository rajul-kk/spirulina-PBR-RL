from twin import loadc, score
import controller as cm
C = loadc('controller.py')
base = score(C); print('base', round(base))
tests = {
 'stir60': dict(stir=60.0), 'stir70': dict(stir=70.0), 'stir90': dict(stir=90.0),
 'burst': dict(burst=(24.0, 2.0, 200.0)), 'burst12': dict(burst=(12.0, 1.0, 200.0)),
 'lmax1700': dict(light_pts=((0.06,600.),(0.2,750.),(0.4,1050.),(0.6,1400.),(0.8,1700.))),
 'lmax2000': dict(light_pts=((0.06,600.),(0.2,750.),(0.4,1050.),(0.6,1400.),(0.8,1700.),(1.2,2000.))),
 'lmax1900': dict(light_pts=((0.06,600.),(0.2,750.),(0.4,1050.),(0.6,1400.),(0.8,1700.),(1.2,1900.))),
 'ramp200': dict(light_up_per_h=200.0), 'ramp50': dict(light_up_per_h=50.0),
 'caps_late': dict(caps=(2.4,)*7 + (2.4, 1.0, 0.0, 0.0)),
 'caps_early': dict(caps=(2.4,)*6 + (2.0, 1.2, 0.5, 0.0, 0.0)),
 'caps3': dict(caps=(3.0,)*7 + (1.7, 0.65, 0.0, 0.0)),
 'caps2': dict(caps=(2.0,)*7 + (1.7, 0.65, 0.0, 0.0)),
}
for k, v in tests.items():
    print(k, round(score(C, v) - base), flush=True)
