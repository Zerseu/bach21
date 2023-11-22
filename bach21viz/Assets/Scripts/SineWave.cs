#region Using

using UnityEngine;

#endregion

[RequireComponent(typeof(AudioSource))]
public sealed class SineWave : MonoBehaviour
{
    public const int Channels = 2;
    public const int SamplingRate = 44100;
    private AudioSource _audioSource;
    private int _position;
    public float Frequency { get; private set; }
    public float Duration { get; private set; }

    private void Start()
    {
        _audioSource = GetComponent<AudioSource>();
    }

    public void PlayNote(float frequency = 440.0f, float duration = 0.25f)
    {
        _audioSource.Stop();
        _position = 0;
        Frequency = frequency;
        Duration = duration;
        var audioClip = AudioClip.Create($"Note {Frequency} Hz", (int)(Duration * SamplingRate), Channels, SamplingRate,
            false, OnAudioRead);
        _audioSource.PlayOneShot(audioClip);
    }

    private void OnAudioRead(float[] data)
    {
        for (var idx = 0; idx < data.Length; idx++)
        {
            data[idx] = Mathf.Sin(2 * Mathf.PI * Frequency * _position / SamplingRate);
            _position++;
        }
    }
}