#region Using

using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Assertions;

#endregion

[RequireComponent(typeof(AudioSource))]
public sealed class NoteGenerator : MonoBehaviour
{
    private const int Channels = 1;
    private const int SamplingRate = 44100;
    private const double ConcertPitch = 440;
    private const float DefaultDuration = 0.25f;
    private static readonly string[] Notes = { "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#" };

    private static readonly double[] HarmonicStrength =
    {
        1,
        0.263,
        0.14,
        0.099,
        0.209,
        0.02,
        0.029,
        0.077,
        0.017,
        0.01
    };

    private static readonly PerlinNoise Noise = new();
    private AudioSource _audioSource;
    private float _frequency, _duration;
    private string[] _notes;
    private int _position;

    public AnimationCurve NoteStrength;


    private static double Note2Freq(string note)
    {
        string[] notes = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
        const double a4 = 440.0;
        const int a4Index = 9;
        const int a4Octave = 4;
        var noteLetter = note[..^1];
        var noteOctave = int.Parse(note[^1].ToString());
        var noteIndex = Array.IndexOf(notes, noteLetter);
        var semitoneDistance = (noteOctave - a4Octave) * 12.0 + (noteIndex - a4Index);
        return a4 * Math.Pow(2, semitoneDistance / 12.0);
    }

    private static string Freq2Note(double freq)
    {
        string[] notes = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
        const double a4 = 440.0;
        const int a4Index = 9;
        const int a4Octave = 4;
        var semitoneDistance = 12.0 * Math.Log(freq / a4) / Math.Log(2.0);
        var totalSemitonesFromC0 = semitoneDistance + (a4Index + a4Octave * 12.0);
        var noteIndex = (int)(Math.Round(totalSemitonesFromC0) % 12);
        var noteOctave = (int)(Math.Round(totalSemitonesFromC0) / 12);
        return notes[noteIndex] + noteOctave;
    }

    private static void RunUnitTest()
    {
        double[] frequency =
        {
            16.35,
            17.32,
            18.35,
            19.45,
            20.60,
            21.83,
            23.12,
            24.50,
            25.96,
            27.50,
            29.14,
            30.87,
            32.70,
            34.65,
            36.71,
            38.89,
            41.20,
            43.65,
            46.25,
            49.00,
            51.91,
            55.00,
            58.27,
            61.74,
            65.41,
            69.30,
            73.42,
            77.78,
            82.41,
            87.31,
            92.50,
            98.00,
            103.83,
            110.00,
            116.54,
            123.47,
            130.81,
            138.59,
            146.83,
            155.56,
            164.81,
            174.61,
            185.00,
            196.00,
            207.65,
            220.00,
            233.08,
            246.94,
            261.63,
            277.18,
            293.66,
            311.13,
            329.63,
            349.23,
            369.99,
            392.00,
            415.30,
            440.00,
            466.16,
            493.88,
            523.25,
            554.37,
            587.33,
            622.25,
            659.26,
            698.46,
            739.99,
            783.99,
            830.61,
            880.00,
            932.33,
            987.77,
            1046.50,
            1108.73,
            1174.66,
            1244.51,
            1318.51,
            1396.91,
            1479.98,
            1567.98,
            1661.22,
            1760.00,
            1864.66,
            1975.53,
            2093.00,
            2217.46,
            2349.32,
            2489.02,
            2637.02,
            2793.83,
            2959.96,
            3135.96,
            3322.44,
            3520.00,
            3729.31,
            3951.07,
            4186.01,
            4434.92,
            4698.64,
            4978.03,
            5274.04,
            5587.65,
            5919.91,
            6271.93,
            6644.88,
            7040.00,
            7458.62,
            7902.13
        };

        foreach (var f in frequency)
            Assert.IsTrue(Math.Abs(Note2Freq(Freq2Note(f)) - f) <= 0.01);
    }

    private void Start()
    {
        RunUnitTest();
        Noise.Initialize();
        _audioSource = GetComponent<AudioSource>();
    }

    private void PlayNote(float frequency, float duration)
    {
        _position = 0;
        _frequency = frequency;
        _duration = duration;
        var audioClip = AudioClip.Create($"Note {_frequency} Hz", (int)(_duration * SamplingRate), Channels,
            SamplingRate,
            false, OnAudioRead);
        _audioSource.PlayOneShot(audioClip);
    }

    private void PlayNote(string note)
    {
        PlayNote((float)Note2Freq(note), DefaultDuration);
    }

    public void PlayNotes(params string[] notes)
    {
        StopAllCoroutines();
        _audioSource.Stop();
        _notes = notes;
        StartCoroutine(PlayNotesCoroutine());
    }

    private IEnumerator PlayNotesCoroutine()
    {
        foreach (var note in _notes)
        {
            PlayNote(note);
            yield return new WaitForSeconds(DefaultDuration);
        }
    }

    private void OnAudioRead(float[] data)
    {
        for (var idx = 0; idx < data.Length; idx++)
        {
            data[idx] = Harmonic();
            _position++;
        }
    }

    private float Harmonic()
    {
        var time = 1.0 * _position / SamplingRate;
        var signal = 0.0;
        for (var harmonic = 1; harmonic <= HarmonicStrength.Length; harmonic++)
        {
            var value = WaveSine(_frequency * harmonic, time);
            signal += HarmonicStrength[harmonic - 1] * value;
        }

        return NoteStrength.Evaluate((float)time / DefaultDuration) * (float)signal;
    }

    private static double WaveSawtooth(double f, double t)
    {
        return 2.0 * (f * t - Math.Floor(0.5 + f * t));
    }

    private static double WaveSquare(double f, double t)
    {
        return 2.0 * (2.0 * Math.Floor(f * t) - Math.Floor(2.0 * f * t)) + 1.0;
    }


    private static double WaveTriangle(double f, double t)
    {
        return 2.0 * Math.Abs(2 * (f * t - Math.Floor(f * t + 0.5))) - 1.0;
    }


    private static double WaveSine(double f, double t)
    {
        return Math.Sin(2.0 * Math.PI * (f * t));
    }
}