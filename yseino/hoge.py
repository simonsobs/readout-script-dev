import sodetlib.operations.uxm_setup as uxm_setup

uxm_setup.setup_amps(S, cfg)
exec(open('/usr/local/src/pysmurf/scratch/shawn/full_band_response.py').read())

band = 0
S.estimate_phase_delay(0)
S.find_freq(band, tone_power=12)
S.setup_notches(band, tone_power=12, new_master_assignment=True)
S.relock(band, tone_power=12)
S.run_serial_gradient_descent(band)
S.set_feedback_enable(band,1)

frac_pp = 0.2
S.tracking_setup(
        band,reset_rate_khz=4,fraction_full_scale=frac_pp,
        make_plot=True,save_plot=True,show_plot=False,
        channel=S.which_on(band),nsamp=2**18,
        lms_gain=1,lms_freq_hz=None,meas_lms_freq=True,
        feedback_start_frac=0.02,feedback_end_frac=0.98)

