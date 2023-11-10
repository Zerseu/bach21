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
            var geoGraph = graph.ToGeometryGraph();
            Debug.Log(geoGraph.Nodes[0].Center);
        }
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
    public readonly bool IsDirected;
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
        IsDirected = int.Parse(tokensDirected[1]) != 0;
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
            var node = new MyNode(id, label, occurrence, length);
            nodes.Add(id, node);
            Nodes.Add(node);
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

    public GeometryGraph ToGeometryGraph()
    {
        const double r = 1;
        const double l = 1;

        var result = new GeometryGraph();

        var nodes = new Dictionary<int, Node>();
        foreach (var nd in Nodes)
        {
            var node = new Node(CurveFactory.CreateCircle(r, new Point()), nd.Id);
            nodes.Add(nd.Id, node);
            result.Nodes.Add(node);
        }

        foreach (var ed in Edges)
        {
            var edge = new Edge(nodes[ed.Source.Id], nodes[ed.Target.Id]) { Length = l };
            result.Edges.Add(edge);
        }

        var settings = new MdsLayoutSettings();
        LayoutHelpers.CalculateLayout(result, settings, null);
        return result;
    }
}