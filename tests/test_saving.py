from anemone.system import *
import os

def test_receiver_save_gridded():
    rx = Receiver(x=np.linspace(-50,50,5), y=np.linspace(-50,50,5), z=0., grid=True)
    rx.save('test_receiver.json')
    rx2 = Receiver.load('test_receiver.json')
    assert torch.allclose(rx.x, rx2.x), "Receiver x are different after saving"
    assert torch.allclose(rx.y, rx2.y), "Receiver y are different after saving"
    assert torch.allclose(rx.z, rx2.z), "Receiver z are different after saving"
    assert torch.allclose(rx.X, rx2.X), "Receiver X are different after saving"
    assert torch.allclose(rx.Y, rx2.Y), "Receiver Y are different after saving"
    assert torch.allclose(rx.Z, rx2.Z), "Receiver Z are different after saving"
    assert rx==rx2, "Receivers are not equal after saving"

    os.remove('test_receiver.json')

def test_receiver_save_not_gridded():
    rx = Receiver(x=np.linspace(-50,50,5), y=np.linspace(-50,50,5), z=0., grid=False)
    rx.save('test_receiver.json')
    rx2 = Receiver.load('test_receiver.json')
    print(rx)
    print(rx2)
    assert torch.allclose(rx.x, rx2.x), "Receiver x are different after saving"
    assert torch.allclose(rx.y, rx2.y), "Receiver y are different after saving"
    assert torch.allclose(rx.z, rx2.z), "Receiver z are different after saving"
    assert torch.allclose(rx.X, rx2.X), "Receiver X are different after saving"
    assert torch.allclose(rx.Y, rx2.Y), "Receiver Y are different after saving"
    assert torch.allclose(rx.Z, rx2.Z), "Receiver Z are different after saving"

    assert rx==rx2, "Receivers are not equal after saving"

    os.remove('test_receiver.json')



def test_loop_save():
    lp = Loop.circle(radius=10., n=20)
    lp.save('test_loop.json', binary_threshold=100)
    lp2 = Loop.load('test_loop.json')
    assert torch.allclose(lp.x, lp2.x), "Loop x are different after saving"
    assert torch.allclose(lp.y, lp2.y), "Loop y are different after saving"
    assert torch.allclose(lp.z, lp2.z), "Loop z are different after saving"
    assert lp==lp2
    # os.remove('test_loop.json')

def test_loop_inequality():
    lp1 = Loop.circle(radius=10., n=20)
    lp2 = Loop.circle(radius=15., n=20)
    assert lp1 != lp2, "Different loops are considered equal"


def test_filterchain_save():
    f1 = ButterworthFilter(1000., order=2)
    f2 = ButterworthFilter(5000., order=1)
    fc = FilterChain([f1, f2])

    fc.save('test_filterchain.json')
    fc2 = FilterChain.load('test_filterchain.json', load_subclasses=True)
    assert len(fc)==len(fc2), "FilterChains have different lengths after saving"
    assert fc==fc2, "FilterChains are not equal after saving"
    os.remove('test_filterchain.json')
        
def test_forward_save():
    from anemone.forwards import Forward

    fwr = Forward.example(tolerance=1e-6)

    rho = torch.tensor([10., 20., 30.])
    thk = torch.tensor([5., 10.])
    dbdt = fwr(rho, thk)

    fwr.save('test_forward.json')

    fwr2 = Forward.load('test_forward.json', load_subclasses=True)

    assert fwr==fwr2, "Forwards are not equal after saving"


    dbdt2 = fwr2(rho, thk)
    assert torch.allclose(dbdt, dbdt2), "Forward responses are different after saving"

    os.remove('test_forward.json')


def test_model_save():
    from anemone.inversion.models import BaseModel
    #res = LogResistivity.smooth(torch.ones(31)*50, torch.logspace(0,2.2,30))
    res = BaseModel([torch.tensor([20,20, 20]), torch.tensor([10., 20.])], names=['resistivity', 'thickness'])
    res.save('test_model.json')
    res2 = BaseModel.load('test_model.json')
    os.remove('test_model.json')
    assert res==res2, "Models are not equal after saving"
    



if __name__ == '__main__':
    # test_loop_save()
    # test_filterchain_save()
    # test_forward_save()
    # test_model_save()