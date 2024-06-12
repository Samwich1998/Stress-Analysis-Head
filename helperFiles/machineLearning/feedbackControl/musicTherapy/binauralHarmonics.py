import os
from pydub import AudioSegment
from pydub.generators import Sine


class binauralBeatsAdjustment:

    def __init__(self, base_frequency_choice, brain_wave_states_choices, left_sound_name, right_sound_name, path_to_wave_files):
        # ------------ for stereo sound overlay generation ------------
        self.base_path = path_to_wave_files
        self.sound_left = AudioSegment.from_file(os.path.join(self.base_path, left_sound_name))
        self.sound_right = AudioSegment.from_file(os.path.join(self.base_path, right_sound_name))
        # -------------------------------------------------------------
        self.base_frequency = base_frequency_choice
        self.brain_wave_states = brain_wave_states_choices
        self.brain_wave_states_harmonic = {'low_delta': 1, 'mid_delta': 2, 'high_delta': 3, 'low_theta': 4, 'high_theta': 6, 'low_alpha': 8,
                                           'high_alpha': 12, 'low_beta': 16, 'high_beta': 24, 'gamma': 32, 'large': 200}

        """ crown: connection with the divine, third_eye: intuition, throat: communication, heart: love, solar: power, sacral: creativity, root: survival"""
        self.base_frequency_name = {'crown': 963, 'third_eye': 852, 'throat': 741, 'heart': 639, 'solar': 528, 'sacral': 417, 'root': 396}

    def get_base_frequency(self, base_frequency_name):
        assert base_frequency_name in self.base_frequency_name, f"Base frequency name {base_frequency_name} not found in the dict"
        return self.base_frequency_name[base_frequency_name]

    def set_brain_wave_frequency_harmonic(self, brain_wave_state):
        assert brain_wave_state in self.brain_wave_states_harmonic, f"Brain wave state {brain_wave_state} not found in the dict"
        return self.brain_wave_states_harmonic[brain_wave_state]

    @staticmethod
    def create_sine_wave_tone(frequency, duration_ms, sample_rate=44100, volume=-20):
        """
        :param frequency: The frequency of the sine wave (in Hz).
        :param duration_ms: The duration of the tone (in milliseconds).
        :param sample_rate: The sample rate for the audio (default to 44100 Hz).
        :param volume: The volume adjustment for the tone (default to -20 dB).
        :return: An AudioSegment containing the sine wave tone.
        """
        # Generate the sine wave tone
        tone = Sine(frequency, sample_rate=sample_rate).to_audio_segment(duration=duration_ms)
        # Adjust the volume of the tone
        tone = tone + volume
        return tone

    def create_binaural_beats(self, base_freq_name, brain_wave_state_harmonic_name, duration_ms=60000, volume=-20):
        base_frequency = self.get_base_frequency(base_freq_name)
        tone_left_pan = AudioSegment.silent(duration=duration_ms)
        tone_right_pan = AudioSegment.silent(duration=duration_ms)
        for brain_wave_state in brain_wave_state_harmonic_name:
            beat_frequency = self.set_brain_wave_frequency_harmonic(brain_wave_state)
            # assume left has the base frequency, right has the base frequency + beat frequency
            frequency_left = base_frequency
            frequency_right = base_frequency + beat_frequency

            # create left and right tones
            tone_left = self.create_sine_wave_tone(frequency_left, duration_ms, volume=volume)
            tone_right = self.create_sine_wave_tone(frequency_right, duration_ms, volume=volume)
            tone_left_pan = tone_left.pan(-1)  # Pan to left
            tone_right_pan = tone_right.pan(1)  # Pan to right

        binaural_beats = tone_left_pan.overlay(tone_right_pan)

        return binaural_beats

    # --------------------------------- for generating stereo sound ---------------------------------
    def create_combined_stereo_beats(self, base_freq_name, brain_wave_state_harmonics, duration_ms=60000, volume=-20):
        base_frequency = self.get_base_frequency(base_freq_name)

        combined_left = AudioSegment.silent(duration=duration_ms)
        combined_right = AudioSegment.silent(duration=duration_ms)

        for brain_wave_state_harmonic_name in brain_wave_state_harmonics:
            beat_frequency = self.set_brain_wave_frequency_harmonic(brain_wave_state_harmonic_name)
            frequency_left = base_frequency
            frequency_right = base_frequency + beat_frequency

            # Create the left and right sine wave tones
            tone_left = self.create_sine_wave_tone(frequency_left, duration_ms, volume=volume)
            tone_right = self.create_sine_wave_tone(frequency_right, duration_ms, volume=volume)

            # Overlay the tones onto the combined audio segments
            combined_left = combined_left.overlay(tone_left)
            combined_right = combined_right.overlay(tone_right)

        # Combine the left and right channels into a stereo audio segment
        binaural_beat = AudioSegment.from_mono_audiosegments(combined_left, combined_right) + volume

        return binaural_beat

    def sound_overlay(self, base_frequency, beat_frequency, white_noise_volume=-20):
        frequency_left = base_frequency
        frequency_right = base_frequency + beat_frequency

        # Generate tones for the given frequencies
        tone_left = self.create_sine_wave_tone(frequency_left, duration_ms=1000)
        tone_right = self.create_sine_wave_tone(frequency_right, duration_ms=1000)

        # combining with white noises
        combined_left = self.sound_left.overlay(tone_left) + white_noise_volume
        combined_right = self.sound_right.overlay(tone_right) + white_noise_volume

        # Pan the combined sounds
        combined_left = combined_left.pan(-1)  # Pan to left
        combined_right = combined_right.pan(1)  # Pan to right

        # Combine the two channels into one stereo sound
        combined_stereo = combined_left.overlay(combined_right) # binaural beats

        return combined_stereo

    def sound_overlay_stereo_beats(self, binaural_beats, white_noise_volume=-10):
        single_combined_stereo = self.sound_left.overlay(binaural_beats) + white_noise_volume
        double_combined_stereo = self.sound_right.overlay(single_combined_stereo) + white_noise_volume

        return double_combined_stereo
    # -------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wave_samples_dir = os.path.join(current_dir, 'Wave_samples')

    # Create directory if it doesn't exist
    if not os.path.exists(wave_samples_dir):
        os.makedirs(wave_samples_dir)

    left_sound = "rain-in-forest.wav"  # file names
    right_sound = "nature-birds.wav"  # file names

    generator = binauralBeatsAdjustment(
        base_frequency_choice=None,
        brain_wave_states_choices=None,
        left_sound_name='rain-in-forest.wav',
        right_sound_name='nature-birds.wav',
        path_to_wave_files=wave_samples_dir
    )

    base_freq_name = 'sacral'  # Base frequency name
    brain_wave_state_harmonics = ['low_alpha', 'gamma', 'large']  # Multiple brain wave state harmonics
    duration_ms = 30000  # Duration units: ms
    volume = -20  # Volume adjustment : dB decreasing

    # Create combined binaural beats
    binaural_beat = generator.create_binaural_beats(base_freq_name, brain_wave_state_harmonics, duration_ms, volume)
    sound_overlay = generator.sound_overlay_stereo_beats(binaural_beat, white_noise_volume=10)

    binaural_beat.export("binaural_beats3.wav", format="wav")
    sound_overlay.export("combined_binaural_beat_with_sound_overlay.wav", format="wav")