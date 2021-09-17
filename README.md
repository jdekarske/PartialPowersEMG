# PartialPowersEMG
 
Previous work [1] in showed that a subject can learn to modulate the powers in specified frequency bands of a surface myographic (sEMG) signal to obtain 2 degree of freedom (DOF) control of a cursor on screen (Figure 1).  This is done by integrating the power within a band and converting that value to a vertical or horizontal displace ment of the cursor. In that work, subjects were asked to move the cursor to one of three target circles and did  so with a 70-98% hit rate. Similar experiments have shown useful for those who have degraded or no motor  control in the lower body because it doesn’t depend on the level of muscular exertion alone, so smaller muscles can be used without fatigue[2]–[5]. However, these frequency bands must be specified offline manually and has  only been tested successfully with two DOF. This work aims to increase the number of and optimize the position of the frequency bands to provide more DOF to the user. To test the efficacy of the system, a subject completed a cursor control task with 2 DOF and the autocorrelation of the signal is examined. Then the system will be scaled to arbitrarily many DOF. The sEMG signal in the time domain only captures the gross sum of voltages of a muscle signal in the sensor’s area which is subject to skin resistance and capacitance [6]. This is a tradeoff with the more invasive needle or fine wire EMG which penetrates and directly measures the muscle voltage. For non-clinical human-machine-interfaces, sEMG is more  tractable to the user and sensor products won’t be subject  to Food and Drug Administration scrutiny because of the  large precedent of substantially equivalent devices [7]. For  these reasons, sEMG is largely preferable to more invasive  procedures for everyday products and motivates this  method. 

sEMG does not guarantee the measurement of a specific muscle because of electrical crosstalk from muscles in  the area [8]. Therefore, we can analyze an sEMG signal with the assumption that itis some combination of signals  from all motor units within the sensor’s range (see sec. Future Work for clarification here). By taking the Fourier  Transform we decompose the signal into its constituent frequencies which represent the magnitudes of each  motor unit. Spikes in the resulting power spectrum would indicate a high amplitude signal in that frequency.  Thus, a motor unit, or collection of motor units, that fire at a certain frequency and with larger amplitude than  their neighbors would exhibit a spike at that frequency. Although motor units are typically recruited according  to the ‘size principle’ to modulate force output, units can be modulated in separate muscles to produce complex movements [9]. For example, Broman et al. show that muscle reflex pathways inhibit individual motor unit ac tivity [10]. The frequency domain of sEMG has yet to be studied in depth, with the focus of research being multi sensor arrays with machine learned classifications [11]–[13]. Adding more sensors increases complexity and cost,  so a simpler solution with less sensors is preferred [14]. We hypothesize that placing an sEMG sensor near a  surface region with densely innervated motor units would allow us to monitor many muscle signals with one  sensor, rather than one sensor for each muscle. 

## Experiment

For this pilot study a single subject performed all trials. The subject is well-versed in EMG control but has not conducted this experiment in full. Since the long-term purpose of this work is to integrate hand prostheses, we  attach a sensor (Delsys Trigno sEMG) to a skin location which would likely bear a prosthetic socket. The flexor  digitorum superficialis is an ideal muscle candidate due to its complex control of individual fingers and therefore  diverse innervation [15]. The sensor records at 2000hz with a range of 11mV and proprietary filtering to correct motion artefacts and stray electrical noise [16]. The sensor is placed above the flexor digitorum superficialis which is identified by flexing, the site is then marked with ink for future experiments. The subject can monitor the exertion of the muscle to ensure proper placement  before the experiment begins. 

The test is composed of a max exertion phase and a control phase. In the max exertion phase, the subject is  instructed to use a cylindrical grip, and maximally contract their hand around a 3 inch diameter PVC pipe for 3  seconds. During this time, the sEMG readings are fed through the processing pipeline. The maximum values from each band are recorded. This is repeated before every control phase, so fatigue effects are controlled. The control phase consists of blocks of 4 trials: a trial for each combination of low and high target cursors (figure  2). These trials are randomized within each block so that after an arbitrary number of blocks, each trial combination appears an equal amount of times. Here, 10 blocks at a time are run before a max exertion phase is run. The trial is successful if the subject overlaps both cursors with the target cursors at the same time and fails after a 10 second timeout.
