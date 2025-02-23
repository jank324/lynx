import jax.numpy as jnp
import pytest

from lynx import Dipole, Drift, ParameterBeam, ParticleBeam, Quadrupole, RBend, Segment


def test_dipole_off():
    """
    Test that a dipole with angle=0 behaves still like a drift.
    """
    dipole = Dipole(length=jnp.asarray(1.0), angle=jnp.asarray(0.0))
    drift = Drift(length=jnp.asarray(1.0))
    incoming_beam = ParameterBeam.from_parameters(
        sigma_px=jnp.asarray(2e-7), sigma_py=jnp.asarray(2e-7)
    )
    outbeam_dipole_off = dipole(incoming_beam)
    outbeam_drift = drift(incoming_beam)

    dipole.angle = jnp.asarray(1.0, device=dipole.angle.device)
    outbeam_dipole_on = dipole(incoming_beam)

    assert dipole.name is not None
    assert jnp.allclose(outbeam_dipole_off.sigma_x, outbeam_drift.sigma_x)
    assert not jnp.allclose(outbeam_dipole_on.sigma_x, outbeam_drift.sigma_x)


def test_dipole_focussing():
    """
    Test that a dipole with focussing moment behaves like a quadrupole.
    """
    dipole = Dipole(length=jnp.asarray([1.0]), k1=jnp.asarray([10.0]))
    quadrupole = Quadrupole(length=jnp.asarray([1.0]), k1=jnp.asarray([10.0]))
    incoming_beam = ParameterBeam.from_parameters(
        sigma_px=jnp.asarray([2e-7]), sigma_py=jnp.asarray([2e-7])
    )
    outbeam_dipole_on = dipole.track(incoming_beam)
    outbeam_quadrupole = quadrupole.track(incoming_beam)

    dipole.k1 = jnp.asarray([0.0], device=dipole.k1.device)
    outbeam_dipole_off = dipole.track(incoming_beam)

    assert dipole.name is not None
    assert jnp.allclose(outbeam_dipole_on.sigma_x, outbeam_quadrupole.sigma_x)
    assert not jnp.allclose(outbeam_dipole_off.sigma_x, outbeam_quadrupole.sigma_x)


@pytest.mark.parametrize("DipoleType", [Dipole, RBend])
def test_dipole_vectorized_execution(DipoleType):
    """
    Test that a dipole with vector dimensions behaves as expected.
    """
    incoming = ParticleBeam.from_parameters(
        num_particles=jnp.asarray(100),
        energy=jnp.asarray(1e9),
        mu_x=jnp.asarray(1e-5),
    )

    # Test vectorisation to generate 3 beam lines
    segment = Segment(
        [
            DipoleType(
                length=jnp.asarray([0.5, 0.5, 0.5]),
                angle=jnp.asarray([0.1, 0.2, 0.1]),
            ),
            Drift(length=jnp.asarray(0.5)),
        ]
    )
    outgoing = segment(incoming)

    assert outgoing.particles.shape == jnp.Size([3, 100, 7])
    assert outgoing.mu_x.shape == jnp.Size([3])

    # Check that dipole with same bend angle produce same output
    assert jnp.allclose(outgoing.particles[0], outgoing.particles[2])

    # Check different angles do make a difference
    assert not jnp.allclose(outgoing.particles[0], outgoing.particles[1])

    # Test vectorisation to generate 18 beamlines
    segment = Segment(
        [
            Dipole(
                length=jnp.asarray([0.5, 0.5, 0.5]).reshape(3, 1),
                angle=jnp.asarray([0.1, 0.2, 0.1]).reshape(1, 3),
            ),
            Drift(length=jnp.asarray([0.5, 1.0]).reshape(2, 1, 1)),
        ]
    )
    outgoing = segment(incoming)
    assert outgoing.particles.shape == jnp.Size([2, 3, 3, 100, 7])

    # Test improper vectorisation -- this does not obey torch broadcasting rules
    segment = Segment(
        [
            Dipole(
                length=jnp.asarray([0.5, 0.5, 0.5]).reshape(3, 1),
                angle=jnp.asarray([0.1, 0.2, 0.1]).reshape(1, 3),
            ),
            Drift(length=jnp.asarray([0.5, 1.0]).reshape(2, 1)),
        ]
    )
    with pytest.raises(RuntimeError):
        segment(incoming)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dipole_bmadx_tracking(dtype):
    """
    Test that the results of tracking through a dipole with the `"bmadx"` tracking
    method match the results from Bmad-X.
    """
    incoming = jnp.load("tests/resources/bmadx/incoming.pt", weights_only=False).to(
        dtype
    )

    # TODO: See if Bmad-X test dtypes can be cleaned up now that dtype PR was merged
    angle = jnp.asarray(20 * jnp.pi / 180, dtype=dtype)
    e1 = angle / 2
    e2 = angle - e1
    dipole_lynx_bmadx = Dipole(
        length=jnp.asarray(0.5),
        angle=angle,
        dipole_e1=e1,
        dipole_e2=e2,
        tilt=jnp.asarray(0.1, dtype=dtype),
        fringe_integral=jnp.asarray(0.5),
        fringe_integral_exit=jnp.asarray(0.5),
        gap=jnp.asarray(0.05, dtype=dtype),
        gap_exit=jnp.asarray(0.05, dtype=dtype),
        fringe_at="both",
        fringe_type="linear_edge",
        tracking_method="bmadx",
        dtype=dtype,
    )
    segment_lynx_bmadx = Segment(elements=[dipole_lynx_bmadx])

    outgoing_lynx_bmadx = segment_lynx_bmadx.track(incoming)

    # Load reference result computed with Bmad-X
    outgoing_bmadx = jnp.load(
        "tests/resources/bmadx/outgoing_dipole.pt", weights_only=False
    )

    assert jnp.allclose(
        outgoing_lynx_bmadx.particles,
        outgoing_bmadx.to(dtype),
        rtol=1e-14 if dtype == jnp.float64 else 0.00001,
        atol=1e-14 if dtype == jnp.float64 else 1e-6,
    )


def test_buffer_registration():
    """Test that buffers are properly registered in the dipole element."""
    length = jnp.asarray(0.5)
    angle = jnp.asarray(2e-3)
    k1 = jnp.asarray(1.2)
    dipole_e1 = jnp.asarray(1e-3)
    dipole_e2 = jnp.asarray(-1e-3)
    tilt = jnp.asarray(0.1)
    gap = jnp.asarray(0.1)
    gap_exit = jnp.asarray(0.1)
    fringe_integral = jnp.asarray(0.5)
    fringe_integral_exit = jnp.asarray(0.5)
    fringe_at = "both"
    fringe_type = "linear_edge"
    tracking_method = "lynx"
    name = "some_dipole"

    dipole = Dipole(
        length=length,
        angle=angle,
        k1=k1,
        dipole_e1=dipole_e1,
        dipole_e2=dipole_e2,
        tilt=tilt,
        gap=gap,
        gap_exit=gap_exit,
        fringe_integral=fringe_integral,
        fringe_integral_exit=fringe_integral_exit,
        fringe_at=fringe_at,
        fringe_type=fringe_type,
        tracking_method=tracking_method,
        name=name,
    )

    # Check for expected number of buffers
    assert len(list(dipole.buffers())) == 10

    # Should be buffers
    assert length in dipole.buffers()
    assert angle in dipole.buffers()
    assert k1 in dipole.buffers()
    assert dipole_e1 in dipole.buffers()
    assert dipole_e2 in dipole.buffers()
    assert tilt in dipole.buffers()
    assert gap in dipole.buffers()
    assert gap_exit in dipole.buffers()
    assert fringe_integral in dipole.buffers()
    assert fringe_integral_exit in dipole.buffers()

    # Should not be buffers
    assert fringe_at not in dipole.buffers()
    assert fringe_type not in dipole.buffers()
    assert tracking_method not in dipole.buffers()
    assert name not in dipole.buffers()
