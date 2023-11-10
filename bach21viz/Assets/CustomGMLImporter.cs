using System.Collections.Generic;
using System.IO;
using Microsoft.Msagl.Core.Geometry;
using Microsoft.Msagl.Core.Geometry.Curves;
using Microsoft.Msagl.Core.Layout;
using Microsoft.Msagl.Layout.MDS;
using Microsoft.Msagl.Miscellaneous;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;

public static class CustomGMLImporter
{
    [MenuItem("Assets/Import *.GML")]
    private static void Import()
    {
        var path = EditorUtility.OpenFilePanel("Graph Selector", Application.dataPath, "gml");
        if (!string.IsNullOrEmpty(path))
        {
            var root = new GameObject("GML Root")
            {
                transform =
                {
                    localPosition = Vector3.zero,
                    localRotation = Quaternion.identity,
                    localScale = Vector3.one
                }
            };

            var graph = new MyGraph(path);
            Debug.Log(graph.IsDirected);
            Debug.Log(graph.Nodes.Count);
            Debug.Log(graph.Edges.Count);

            Create();
        }
    }

    private static GeometryGraph Create()
    {
        const double r = 10;
        const double l = 100;

        var graph = new GeometryGraph();
        var a = new Node(CurveFactory.CreateCircle(r, new Point()), "a");
        var b = new Node(CurveFactory.CreateCircle(r, new Point()), "b");
        var c = new Node(CurveFactory.CreateCircle(r, new Point()), "c");
        var d = new Node(CurveFactory.CreateCircle(r, new Point()), "d");
        graph.Nodes.Add(a);
        graph.Nodes.Add(b);
        graph.Nodes.Add(c);
        graph.Nodes.Add(d);

        graph.Edges.Add(new Edge(a, b) { Length = l });
        graph.Edges.Add(new Edge(b, c) { Length = l });
        graph.Edges.Add(new Edge(b, d) { Length = l });

        var settings = new MdsLayoutSettings();
        LayoutHelpers.CalculateLayout(graph, settings, null);
        return graph;
    }
}

public sealed class MyNode
{
    public MyNode(int id, string label, int occurrence, int length)
    {
        Id = id;
        Label = label;
        Occurrence = occurrence;
        Length = length;
    }

    public int Id { get; }

    public string Label { get; }
    public int Length { get; }
    public int Occurrence { get; }
}

public sealed class MyEdge
{
    public MyEdge(MyNode source, MyNode target)
    {
        Source = source;
        Target = target;
    }

    public MyNode Source { get; }
    public MyNode Target { get; }
}

public sealed class MyGraph
{
    public readonly List<MyEdge> Edges = new();
    public readonly bool IsDirected = false;
    public readonly List<MyNode> Nodes = new();

    public MyGraph(string gmlPath)
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
        var isDirected = int.Parse(tokensDirected[1]) != 0;
        gmlIndex++;

        var nodes = new Dictionary<int, MyNode>();
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
            var vertex = new MyNode(id, label, occurrence, length);
            nodes.Add(id, vertex);
            Nodes.Add(vertex);
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
            Edges.Add(new MyEdge(nodes[source], nodes[target]));
        }

        Assert.IsTrue(gmlContent[gmlIndex].Trim() == "]");
    }
}