using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;

public static class CustomGMLImporter
{
    [MenuItem("Assets/Import *.GML")]
    private static void Import()
    {
        var gmlPath = EditorUtility.OpenFilePanel("Graph Selector", Application.dataPath, "gml");
        if (!string.IsNullOrEmpty(gmlPath))
        {
            var gmlRoot = new GameObject("GML Root")
            {
                transform =
                {
                    localPosition = Vector3.zero,
                    localRotation = Quaternion.identity,
                    localScale = Vector3.one
                }
            };

            var gml = new Graph(gmlPath);
            Debug.Log(gml.Directed);
            Debug.Log(gml.Nodes.Count);
            Debug.Log(gml.Edges.Count);
        }
    }
}

public readonly struct Node
{
    public readonly int Id;
    public readonly string Label;
    public readonly int Occurrence;
    public readonly int Length;

    public Node(int id, string label, int occurrence, int length)
    {
        Id = id;
        Label = label;
        Occurrence = occurrence;
        Length = length;
    }
}

public readonly struct Edge
{
    public readonly int Source;
    public readonly int Target;

    public Edge(int source, int target)
    {
        Source = source;
        Target = target;
    }
}

public sealed class Graph
{
    public readonly bool Directed;
    public readonly List<Edge> Edges = new();
    public readonly List<Node> Nodes = new();

    public Graph(string gmlPath)
    {
        var gmlContent = File.ReadAllLines(gmlPath);

        var gmlIndex = 0;
        while (gmlContent[gmlIndex].Trim() != "graph")
            gmlIndex++;
        gmlIndex++;
        Assert.IsTrue(gmlContent[gmlIndex].Trim() == "[");
        gmlIndex++;
        var tokensDirected = gmlContent[gmlIndex].Trim().Split();
        Assert.IsTrue(tokensDirected[0] == "directed");
        Directed = int.Parse(tokensDirected[1]) != 0;
        gmlIndex++;

        while (gmlContent[gmlIndex].Trim() == "node")
        {
            gmlIndex++;
            Assert.IsTrue(gmlContent[gmlIndex].Trim() == "[");
            gmlIndex++;
            var tokensId = gmlContent[gmlIndex].Trim().Split();
            Assert.IsTrue(tokensId[0] == "id");
            var id = int.Parse(tokensId[1]);
            gmlIndex++;
            var tokensLabel = gmlContent[gmlIndex].Trim().Split();
            Assert.IsTrue(tokensLabel[0] == "label");
            var label = tokensLabel[1].Trim('"');
            gmlIndex++;
            var tokensOccurrence = gmlContent[gmlIndex].Trim().Split();
            Assert.IsTrue(tokensOccurrence[0] == "occurrence");
            var occurrence = int.Parse(tokensOccurrence[1]);
            gmlIndex++;
            var tokensLength = gmlContent[gmlIndex].Trim().Split();
            Assert.IsTrue(tokensLength[0] == "length");
            var length = int.Parse(tokensLength[1]);
            gmlIndex++;
            Assert.IsTrue(gmlContent[gmlIndex].Trim() == "]");
            gmlIndex++;
            Nodes.Add(new Node(id, label, occurrence, length));
        }

        while (gmlContent[gmlIndex].Trim() == "edge")
        {
            gmlIndex++;
            Assert.IsTrue(gmlContent[gmlIndex].Trim() == "[");
            gmlIndex++;
            var tokensSource = gmlContent[gmlIndex].Trim().Split();
            Assert.IsTrue(tokensSource[0] == "source");
            var source = int.Parse(tokensSource[1]);
            gmlIndex++;
            var tokensTarget = gmlContent[gmlIndex].Trim().Split();
            Assert.IsTrue(tokensTarget[0] == "target");
            var target = int.Parse(tokensTarget[1]);
            gmlIndex++;
            Assert.IsTrue(gmlContent[gmlIndex].Trim() == "]");
            gmlIndex++;
            Edges.Add(new Edge(source, target));
        }

        Assert.IsTrue(gmlContent[gmlIndex].Trim() == "]");
    }
}