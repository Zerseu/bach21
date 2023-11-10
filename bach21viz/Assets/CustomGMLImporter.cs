using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Msagl.Core.Geometry;
using Microsoft.Msagl.Core.Geometry.Curves;
using Microsoft.Msagl.Core.Layout;
using Microsoft.Msagl.Layout.Layered;
using Microsoft.Msagl.Miscellaneous;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;

public static class CustomGMLImporter
{
    [MenuItem("Assets/Import *.GML")]
    private static void Import()
    {
        var path = EditorUtility.OpenFilePanel("Layout Selector", Application.dataPath, "gml");
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
            foreach (var node in graph.Layout.Nodes)
            {
                var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere.transform.parent = root.transform;
                sphere.transform.localPosition =
                    new Vector3((float)(node.Center.X / 10.0), 0, (float)(node.Center.Y / 10.0));
                sphere.transform.localRotation = Quaternion.identity;
                sphere.transform.localScale = Vector3.one;
            }
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
    public const float NodeRadius = 1f;
    public const float EdgeLength = 1f;
    public readonly List<MyEdge> Edges = new();
    public readonly bool IsDirected;
    public readonly GeometryGraph Layout = new();
    public readonly Dictionary<int, Tuple<Node, MyNode>> Map = new();
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
            var node = new Node(CurveFactory.CreateCircle(NodeRadius, new Point()), id);
            var myNode = new MyNode(id, label, occurrence, length);
            Map.Add(id, new Tuple<Node, MyNode>(node, myNode));
            Layout.Nodes.Add(node);
            Nodes.Add(myNode);
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
            Layout.Edges.Add(new Edge(Map[source].Item1, Map[target].Item1) { Length = EdgeLength });
            Edges.Add(new MyEdge(Map[source].Item2, Map[target].Item2));
        }

        Assert.IsTrue(gmlContent[gmlIndex].Trim() == "]");
        LayoutHelpers.CalculateLayout(Layout, new SugiyamaLayoutSettings(), null);
    }
}