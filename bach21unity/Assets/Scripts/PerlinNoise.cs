#region Using

using System;
using System.Collections.Generic;
using UnityEngine;
using Random = System.Random;

#endregion

internal static class PerlinNoiseHelpers
{
    internal static byte[] AsByte<T>(this T[] input, Func<T, byte[]> convert)
    {
        var ret = new List<byte>();
        foreach (var item in input)
            ret.AddRange(convert(item));
        return ret.ToArray();
    }

    internal static T[] Flatten<T>(this T[,] matrix)
    {
        var HEIGHT = matrix.GetLength(0);
        var WIDTH = matrix.GetLength(1);
        var ret = new T[HEIGHT * WIDTH];

        for (var i = 0; i < HEIGHT; i++)
        for (var j = 0; j < WIDTH; j++)
            ret[j + i * WIDTH] = matrix[i, j];
        return ret;
    }
}

/// <summary>
///     Class that exposes functionality for Perlin noise generation (1D, 2D, 3D and 4D noise).
/// </summary>
public sealed class PerlinNoise
{
    public void Initialize(int seed = 42)
    {
        //Create a random number generator with the specified seed...
        var random = new Random(seed);

        //Reset permutation table...
        var indexVisited = new bool[256];
        Array.Clear(indexVisited, 0, 256);
        Array.Clear(Permutation, 0, 256);

        //Generate a random permutation...
        for (byte i = 0; i <= 255; i++)
        {
            while (true)
            {
                var index = (byte)random.Next();
                if (!indexVisited[index])
                {
                    Permutation[index] = i;
                    indexVisited[index] = true;
                    break;
                }
            }

            if (i == 255)
                break;
        }

        //Construct 2D permutation table...
        for (byte px = 0; px <= 255; px++)
        {
            for (byte py = 0; py <= 255; py++)
            {
                var A = (byte)(Permutation[px] + py);
                var B = (byte)(Permutation[(byte)(px + 1)] + py);

                var AA = Permutation[A];
                var AB = Permutation[(byte)(A + 1)];
                var BA = Permutation[B];
                var BB = Permutation[(byte)(B + 1)];

                Permutation2D[py, px] = new Color32(AA, AB, BA, BB);

                if (py == 255)
                    break;
            }

            if (px == 255)
                break;
        }

        //Construct permutation tables for 1D, 2D, 3D, 4D gradients...
        for (byte p = 0; p <= 255; p++)
        {
            G1Permutation[p] = G1[Permutation[p] & 1];
            G2Permutation[p] = G2[Permutation[p] & 3];
            G3Permutation[p] = G3[Permutation[p] & 15];
            G4Permutation[p] = G4[Permutation[p] & 31];

            if (p == 255)
                break;
        }

        //Generate all textures needed for the HLSL effect file...
        if (PermutationTexture == null)
            PermutationTexture = new Texture2D(256, 1, TextureFormat.Alpha8, false, false);
        PermutationTexture.LoadRawTextureData(Permutation.AsByte(p => new[] { p }));
        PermutationTexture.filterMode = FilterMode.Point;
        PermutationTexture.wrapMode = TextureWrapMode.Repeat;
        PermutationTexture.Apply(false, false);

        if (PermutationTexture2D == null)
            PermutationTexture2D = new Texture2D(256, 256, TextureFormat.RGBA32, false, false);
        PermutationTexture2D.LoadRawTextureData(Permutation2D.Flatten().AsByte(p => new[] { p.r, p.g, p.b, p.a }));
        PermutationTexture2D.filterMode = FilterMode.Point;
        PermutationTexture2D.wrapMode = TextureWrapMode.Repeat;
        PermutationTexture2D.Apply();

        if (G3Texture == null)
        {
            G3Texture = new Texture2D(16, 1, TextureFormat.RGB24, false, false);
            G3Texture.LoadRawTextureData(
                G3.AsByte(
                    p =>
                        new[]
                        {
                            (byte)((p.x / 2 + 0.5) * 255), (byte)((p.y / 2 + 0.5) * 255), (byte)((p.z / 2 + 0.5) * 255)
                        }));
            G3Texture.filterMode = FilterMode.Point;
            G3Texture.wrapMode = TextureWrapMode.Repeat;
            G3Texture.Apply();
        }

        if (G3PermutationTexture == null)
            G3PermutationTexture = new Texture2D(256, 1, TextureFormat.RGB24, false, false);
        G3PermutationTexture.LoadRawTextureData(
            G3Permutation.AsByte(
                p =>
                    new[]
                    {
                        (byte)((p.x / 2 + 0.5) * 255), (byte)((p.y / 2 + 0.5) * 255), (byte)((p.z / 2 + 0.5) * 255)
                    }));
        G3PermutationTexture.filterMode = FilterMode.Point;
        G3PermutationTexture.wrapMode = TextureWrapMode.Repeat;
        G3PermutationTexture.Apply();

        if (G4Texture == null)
        {
            G4Texture = new Texture2D(32, 1, TextureFormat.RGBA32, false, false);
            G4Texture.LoadRawTextureData(
                G4.AsByte(
                    p =>
                        new[]
                        {
                            (byte)((p.x / 2 + 0.5) * 255), (byte)((p.y / 2 + 0.5) * 255), (byte)((p.z / 2 + 0.5) * 255),
                            (byte)((p.w / 2 + 0.5) * 255)
                        }));
            G4Texture.filterMode = FilterMode.Point;
            G4Texture.wrapMode = TextureWrapMode.Repeat;
            G4Texture.Apply();
        }

        if (G4PermutationTexture == null)
            G4PermutationTexture = new Texture2D(256, 1, TextureFormat.RGBA32, false, false);
        G4PermutationTexture.LoadRawTextureData(
            G4Permutation.AsByte(
                p =>
                    new[]
                    {
                        (byte)((p.x / 2 + 0.5) * 255), (byte)((p.y / 2 + 0.5) * 255), (byte)((p.z / 2 + 0.5) * 255),
                        (byte)((p.w / 2 + 0.5) * 255)
                    }));
        G4PermutationTexture.filterMode = FilterMode.Point;
        G4PermutationTexture.wrapMode = TextureWrapMode.Repeat;
        G4PermutationTexture.Apply();
    }

    public void Apply(Material material)
    {
        material.SetTexture("PermutationTexture", PermutationTexture);
        material.SetTexture("PermutationTexture2D", PermutationTexture2D);
        material.SetTexture("G3Texture", G3Texture);
        material.SetTexture("G3PermutationTexture", G3PermutationTexture);
        material.SetTexture("G4Texture", G4Texture);
        material.SetTexture("G4PermutationTexture", G4PermutationTexture);
    }

    #region Properties

    /// <summary>
    ///     Constant epsilon (small value) for computing gradient functions.
    /// </summary>
    public const float EPS = 0.001f;

    /// <summary>
    ///     Permutation table.
    /// </summary>
    public readonly byte[] Permutation = new byte[256];

    /// <summary>
    ///     2D Permutation table.
    /// </summary>
    public readonly Color32[,] Permutation2D = new Color32[256, 256];

    /// <summary>
    ///     Gradients for 1D noise.
    /// </summary>
    public static readonly float[] G1 =
    {
        1,
        -1
    };

    /// <summary>
    ///     Permutation for 1D noise gradients.
    /// </summary>
    public static readonly float[] G1Permutation = new float[256];

    /// <summary>
    ///     Gradients for 2D noise.
    /// </summary>
    public static readonly Vector2[] G2 =
    {
        new(1, 1),
        new(1, -1),
        new(-1, 1),
        new(-1, -1)
    };

    /// <summary>
    ///     Permutation for 2D noise gradients.
    /// </summary>
    public static readonly Vector2[] G2Permutation = new Vector2[256];

    /// <summary>
    ///     Gradients for 3D noise.
    /// </summary>
    public static readonly Vector3[] G3 =
    {
        new(1, 1, 0),
        new(-1, 1, 0),
        new(1, -1, 0),
        new(-1, -1, 0),
        new(1, 0, 1),
        new(-1, 0, 1),
        new(1, 0, -1),
        new(-1, 0, -1),
        new(0, 1, 1),
        new(0, -1, 1),
        new(0, 1, -1),
        new(0, -1, -1),
        new(1, 1, 0),
        new(0, -1, 1),
        new(-1, 1, 0),
        new(0, -1, -1)
    };

    /// <summary>
    ///     Permutation for 3D noise gradients.
    /// </summary>
    public static readonly Vector3[] G3Permutation = new Vector3[256];

    /// <summary>
    ///     Gradients for 4D noise.
    /// </summary>
    public static readonly Vector4[] G4 =
    {
        new(0, -1, -1, -1),
        new(0, -1, -1, 1),
        new(0, -1, 1, -1),
        new(0, -1, 1, 1),
        new(0, 1, -1, -1),
        new(0, 1, -1, 1),
        new(0, 1, 1, -1),
        new(0, 1, 1, 1),
        new(-1, -1, 0, -1),
        new(-1, 1, 0, -1),
        new(1, -1, 0, -1),
        new(1, 1, 0, -1),
        new(-1, -1, 0, 1),
        new(-1, 1, 0, 1),
        new(1, -1, 0, 1),
        new(1, 1, 0, 1),
        new(-1, 0, -1, -1),
        new(1, 0, -1, -1),
        new(-1, 0, -1, 1),
        new(1, 0, -1, 1),
        new(-1, 0, 1, -1),
        new(1, 0, 1, -1),
        new(-1, 0, 1, 1),
        new(1, 0, 1, 1),
        new(0, -1, -1, 0),
        new(0, -1, -1, 0),
        new(0, -1, 1, 0),
        new(0, -1, 1, 0),
        new(0, 1, -1, 0),
        new(0, 1, -1, 0),
        new(0, 1, 1, 0),
        new(0, 1, 1, 0)
    };

    /// <summary>
    ///     Permutation for 4D noise gradients.
    /// </summary>
    public static readonly Vector4[] G4Permutation = new Vector4[256];

    public Texture2D PermutationTexture { get; private set; }
    public Texture2D PermutationTexture2D { get; private set; }
    public static Texture2D G3Texture { get; private set; }
    public Texture2D G3PermutationTexture { get; private set; }
    public static Texture2D G4Texture { get; private set; }
    public Texture2D G4PermutationTexture { get; private set; }

    #endregion

    #region Helper Functions

    public static float fade(float t)
    {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    public static float lerp(float a, float b, float t)
    {
        return a + t * (b - a);
    }

    public static float grad(byte hash, float x)
    {
        var aux = G1[hash & 1];
        return x * aux;
    }

    public static float grad(byte hash, float x, float y)
    {
        var aux = G2[hash & 3];
        return x * aux.x + y * aux.y;
    }

    public static float grad(byte hash, float x, float y, float z)
    {
        var aux = G3[hash & 15];
        return x * aux.x + y * aux.y + z * aux.z;
    }

    public static float grad(byte hash, float x, float y, float z, float w)
    {
        var aux = G4[hash & 31];
        return x * aux.x + y * aux.y + z * aux.z + w * aux.w;
    }

    public float gradperm(byte hash, float x)
    {
        var aux = G1Permutation[hash];
        return x * aux;
    }

    public float gradperm(byte hash, float x, float y)
    {
        var aux = G2Permutation[hash];
        return x * aux.x + y * aux.y;
    }

    public float gradperm(byte hash, float x, float y, float z)
    {
        var aux = G3Permutation[hash];
        return x * aux.x + y * aux.y + z * aux.z;
    }

    public float gradperm(byte hash, float x, float y, float z, float w)
    {
        var aux = G4Permutation[hash];
        return x * aux.x + y * aux.y + z * aux.z + w * aux.w;
    }

    public static float gradient(float x, Func<float, float> f)
    {
        var F0 = f(x);
        var Fx = f(x + EPS);
        return (Fx - F0) / EPS;
    }

    public static Vector2 gradient(float x, float y, Func<float, float, float> f)
    {
        var F0 = f(x, y);
        var Fx = f(x + EPS, y);
        var Fy = f(x, y + EPS);
        return new Vector2(Fx - F0, Fy - F0) / EPS;
    }

    public static Vector3 gradient(float x, float y, float z, Func<float, float, float, float> f)
    {
        var F0 = f(x, y, z);
        var Fx = f(x + EPS, y, z);
        var Fy = f(x, y + EPS, z);
        var Fz = f(x, y, z + EPS);
        return new Vector3(Fx - F0, Fy - F0, Fz - F0) / EPS;
    }

    public static Vector4 gradient(float x, float y, float z, float w, Func<float, float, float, float, float> f)
    {
        var F0 = f(x, y, z, w);
        var Fx = f(x + EPS, y, z, w);
        var Fy = f(x, y + EPS, z, w);
        var Fz = f(x, y, z + EPS, w);
        var Fw = f(x, y, z, w + EPS);
        return new Vector4(Fx - F0, Fy - F0, Fz - F0, Fw - F0) / EPS;
    }

    public static Vector4 gradienttiled(float x, float y, Func<float, float, float, float, float> f)
    {
        return
            Vector4.Normalize(gradient((float)Math.Sin(x), (float)Math.Cos(x), (float)Math.Sin(y),
                (float)Math.Cos(y), f));
    }

    #endregion

    #region Simple Noise Functions

    public float noise(float x)
    {
        //Compute the cell coordinates...
        var X = (byte)Math.Floor(x);

        //Retrieve the decimal part of the cell...
        x -= (float)Math.Floor(x);

        //Smooth the curve...
        var xt = fade(x);

        //Interpolate between directions...
        return lerp(grad(Permutation[X], x), grad(Permutation[(byte)(X + 1)], x - 1), xt);
    }

    public float noise(float x, float y)
    {
        //Compute the cell coordinates...
        var X = (byte)Math.Floor(x);
        var Y = (byte)Math.Floor(y);

        //Retrieve the decimal part of the cell...
        x -= (float)Math.Floor(x);
        y -= (float)Math.Floor(y);

        //Smooth the curve...
        var xt = fade(x);
        var yt = fade(y);

        //Fetch some random values from the table...
        var A = (byte)(Permutation[X] + Y);
        var B = (byte)(Permutation[(byte)(X + 1)] + Y);

        //Interpolate between directions...
        return lerp(lerp(grad(Permutation[A], x, y), grad(Permutation[B], x - 1, y), xt),
            lerp(grad(Permutation[(byte)(A + 1)], x, y - 1), grad(Permutation[(byte)(B + 1)], x - 1, y - 1), xt),
            yt);
    }

    public float noise(float x, float y, float z)
    {
        //Compute the cell coordinates...
        var X = (byte)Math.Floor(x);
        var Y = (byte)Math.Floor(y);
        var Z = (byte)Math.Floor(z);

        //Retrieve the decimal part of the cell...
        x -= (float)Math.Floor(x);
        y -= (float)Math.Floor(y);
        z -= (float)Math.Floor(z);

        //Smooth the curve...
        var xt = fade(x);
        var yt = fade(y);
        var zt = fade(z);

        //Fetch some random values from the table...
        var A = (byte)(Permutation[X] + Y);
        var B = (byte)(Permutation[(byte)(X + 1)] + Y);

        var AA = (byte)(Permutation[A] + Z);
        var AB = (byte)(Permutation[(byte)(A + 1)] + Z);
        var BA = (byte)(Permutation[B] + Z);
        var BB = (byte)(Permutation[(byte)(B + 1)] + Z);

        //Interpolate between directions...
        return
            lerp(
                lerp(lerp(grad(Permutation[AA], x, y, z), grad(Permutation[BA], x - 1, y, z), xt),
                    lerp(grad(Permutation[AB], x, y - 1, z), grad(Permutation[BB], x - 1, y - 1, z), xt), yt),
                lerp(
                    lerp(grad(Permutation[(byte)(AA + 1)], x, y, z - 1),
                        grad(Permutation[(byte)(BA + 1)], x - 1, y, z - 1), xt),
                    lerp(grad(Permutation[(byte)(AB + 1)], x, y - 1, z - 1),
                        grad(Permutation[(byte)(BB + 1)], x - 1, y - 1, z - 1), xt), yt), zt);
    }

    public float noise(float x, float y, float z, float w)
    {
        //Compute the cell coordinates...
        var X = (byte)Math.Floor(x);
        var Y = (byte)Math.Floor(y);
        var Z = (byte)Math.Floor(z);
        var W = (byte)Math.Floor(w);

        //Retrieve the decimal part of the cell...
        x -= (float)Math.Floor(x);
        y -= (float)Math.Floor(y);
        z -= (float)Math.Floor(z);
        w -= (float)Math.Floor(w);

        //Smooth the curve...
        var xt = fade(x);
        var yt = fade(y);
        var zt = fade(z);
        var wt = fade(w);

        //Fetch some random values from the table...
        var A = (byte)(Permutation[X] + Y);
        var B = (byte)(Permutation[(byte)(X + 1)] + Y);

        var AA = (byte)(Permutation[A] + Z);
        var AB = (byte)(Permutation[(byte)(A + 1)] + Z);
        var BA = (byte)(Permutation[B] + Z);
        var BB = (byte)(Permutation[(byte)(B + 1)] + Z);

        var AAA = (byte)(Permutation[AA] + W);
        var AAB = (byte)(Permutation[(byte)(AA + 1)] + W);
        var ABA = (byte)(Permutation[AB] + W);
        var ABB = (byte)(Permutation[(byte)(AB + 1)] + W);
        var BAA = (byte)(Permutation[BA] + W);
        var BAB = (byte)(Permutation[(byte)(BA + 1)] + W);
        var BBA = (byte)(Permutation[BB] + W);
        var BBB = (byte)(Permutation[(byte)(BB + 1)] + W);

        //Interpolate between directions...
        return
            lerp(
                lerp(
                    lerp(lerp(grad(Permutation[AAA], x, y, z, w), grad(Permutation[BAA], x - 1, y, z, w), xt),
                        lerp(grad(Permutation[ABA], x, y - 1, z, w), grad(Permutation[BBA], x - 1, y - 1, z, w), xt),
                        yt),
                    lerp(
                        lerp(grad(Permutation[AAB], x, y, z - 1, w), grad(Permutation[BAB], x - 1, y, z - 1, w), xt),
                        lerp(grad(Permutation[ABB], x, y - 1, z - 1, w),
                            grad(Permutation[BBB], x - 1, y - 1, z - 1, w), xt), yt), zt),
                lerp(
                    lerp(
                        lerp(grad(Permutation[(byte)(AAA + 1)], x, y, z, w - 1),
                            grad(Permutation[(byte)(BAA + 1)], x - 1, y, z, w - 1), xt),
                        lerp(grad(Permutation[(byte)(ABA + 1)], x, y - 1, z, w - 1),
                            grad(Permutation[(byte)(BBA + 1)], x - 1, y - 1, z, w - 1), xt), yt),
                    lerp(
                        lerp(grad(Permutation[(byte)(AAB + 1)], x, y, z - 1, w - 1),
                            grad(Permutation[(byte)(BAB + 1)], x - 1, y, z - 1, w - 1), xt),
                        lerp(grad(Permutation[(byte)(ABB + 1)], x, y - 1, z - 1, w - 1),
                            grad(Permutation[(byte)(BBB + 1)], x - 1, y - 1, z - 1, w - 1), xt), yt), zt), wt);
    }

    #endregion

    #region Improved Noise Functions

    [Obsolete("Please consider using the noise(x) function as it is faster.", false)]
    public float inoise(float x)
    {
        //Compute the cell coordinates...
        var X = (byte)Math.Floor(x);

        //Retrieve the decimal part of the cell...
        x -= (float)Math.Floor(x);

        //Smooth the curve...
        var xt = fade(x);

        //Interpolate between directions...
        return lerp(gradperm(X, x), gradperm((byte)(X + 1), x - 1), xt);
    }

    [Obsolete("Please consider using the noise(x, y) function as it is faster.", false)]
    public float inoise(float x, float y)
    {
        //Compute the cell coordinates...
        var X = (byte)Math.Floor(x);
        var Y = (byte)Math.Floor(y);

        //Retrieve the decimal part of the cell...
        x -= (float)Math.Floor(x);
        y -= (float)Math.Floor(y);

        //Smooth the curve...
        var xt = fade(x);
        var yt = fade(y);

        //Fetch some random values from the table...
        var AA = Permutation2D[Y, X];

        //Interpolate between directions...
        return lerp(lerp(grad(AA.r, x, y), grad(AA.b, x - 1, y), xt),
            lerp(grad(AA.g, x, y - 1), grad(AA.a, x - 1, y - 1), xt), yt);
    }

    [Obsolete("Please consider using the noise(x, y, z) function as it is faster.", false)]
    public float inoise(float x, float y, float z)
    {
        //Compute the cell coordinates...
        var X = (byte)Math.Floor(x);
        var Y = (byte)Math.Floor(y);
        var Z = (byte)Math.Floor(z);

        //Retrieve the decimal part of the cell...
        x -= (float)Math.Floor(x);
        y -= (float)Math.Floor(y);
        z -= (float)Math.Floor(z);

        //Smooth the curve...
        var xt = fade(x);
        var yt = fade(y);
        var zt = fade(z);

        //Fetch some random values from the table...
        var AA = Permutation2D[Y, X];
        AA.r += Z;
        AA.g += Z;
        AA.b += Z;
        AA.a += Z;

        //Interpolate between directions...
        return
            lerp(
                lerp(lerp(gradperm(AA.r, x, y, z), gradperm(AA.b, x - 1, y, z), xt),
                    lerp(gradperm(AA.g, x, y - 1, z), gradperm(AA.a, x - 1, y - 1, z), xt), yt),
                lerp(
                    lerp(gradperm((byte)(AA.r + 1), x, y, z - 1), gradperm((byte)(AA.b + 1), x - 1, y, z - 1), xt),
                    lerp(gradperm((byte)(AA.g + 1), x, y - 1, z - 1),
                        gradperm((byte)(AA.a + 1), x - 1, y - 1, z - 1), xt), yt), zt);
    }

    [Obsolete("Please consider using the noise(x, y, z, w) function as it is faster.", false)]
    public float inoise(float x, float y, float z, float w)
    {
        //Compute the cell coordinates...
        var X = (byte)Math.Floor(x);
        var Y = (byte)Math.Floor(y);
        var Z = (byte)Math.Floor(z);
        var W = (byte)Math.Floor(w);

        //Retrieve the decimal part of the cell...
        x -= (float)Math.Floor(x);
        y -= (float)Math.Floor(y);
        z -= (float)Math.Floor(z);
        w -= (float)Math.Floor(w);

        //Smooth the curve...
        var xt = fade(x);
        var yt = fade(y);
        var zt = fade(z);
        var wt = fade(w);

        //Fetch some random values from the table...
        var A = (byte)(Permutation[X] + Y);
        var B = (byte)(Permutation[(byte)(X + 1)] + Y);

        var AA = Permutation2D[Z, A];
        AA.r += W;
        AA.g += W;
        AA.b += W;
        AA.a += W;
        var BB = Permutation2D[Z, B];
        BB.r += W;
        BB.g += W;
        BB.b += W;
        BB.a += W;

        //Interpolate between directions...
        return
            lerp(
                lerp(
                    lerp(lerp(gradperm(AA.r, x, y, z, w), gradperm(BB.r, x - 1, y, z, w), xt),
                        lerp(gradperm(AA.b, x, y - 1, z, w), gradperm(BB.b, x - 1, y - 1, z, w), xt), yt),
                    lerp(lerp(gradperm(AA.g, x, y, z - 1, w), gradperm(BB.g, x - 1, y, z - 1, w), xt),
                        lerp(gradperm(AA.a, x, y - 1, z - 1, w), gradperm(BB.a, x - 1, y - 1, z - 1, w), xt), yt),
                    zt),
                lerp(
                    lerp(
                        lerp(gradperm((byte)(AA.r + 1), x, y, z, w - 1),
                            gradperm((byte)(BB.r + 1), x - 1, y, z, w - 1), xt),
                        lerp(gradperm((byte)(AA.b + 1), x, y - 1, z, w - 1),
                            gradperm((byte)(BB.b + 1), x - 1, y - 1, z, w - 1), xt), yt),
                    lerp(
                        lerp(gradperm((byte)(AA.g + 1), x, y, z - 1, w - 1),
                            gradperm((byte)(BB.g + 1), x - 1, y, z - 1, w - 1), xt),
                        lerp(gradperm((byte)(AA.a + 1), x, y - 1, z - 1, w - 1),
                            gradperm((byte)(BB.a + 1), x - 1, y - 1, z - 1, w - 1), xt), yt), zt), wt);
    }

    #endregion

    #region Composite Noise Functions

    public float None(float x, float y)
    {
        return 0;
    }

    public float Simple(float x, float y)
    {
        return noise(x, y);
    }

    public float SimpleTiled(float x, float y)
    {
        return noise((float)Math.Sin(x), (float)Math.Cos(x), (float)Math.Sin(y), (float)Math.Cos(y));
    }

    public Vector2 NoneGradient(float x, float y)
    {
        return Vector2.zero;
    }

    public Vector2 SimpleGradient(float x, float y)
    {
        return gradient(x, y, noise).normalized;
    }

    public Vector4 SimpleGradientTiled(float x, float y)
    {
        return gradienttiled(x, y, noise);
    }

    #endregion
}