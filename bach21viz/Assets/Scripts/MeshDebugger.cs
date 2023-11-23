using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(BoxCollider))]
public class MeshDebugger : MonoBehaviour
{
    private Color _color;
    private MeshRenderer _renderer;
    [TextArea(10, 20)] public string Info;

    private void Start()
    {
        _renderer = GetComponent<MeshRenderer>();
        _color = _renderer.material.color;
    }

    private void OnMouseEnter()
    {
        _renderer.material.color = Color.yellow;
    }

    private void OnMouseExit()
    {
        _renderer.material.color = _color;
    }

    private void OnMouseUpAsButton()
    {
        Camera.main.GetComponent<SineWave>().PlayNotes(Info.Split());
    }
}