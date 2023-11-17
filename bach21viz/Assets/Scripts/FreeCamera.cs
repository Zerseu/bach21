using UnityEngine;

public class FreeCamera : MonoBehaviour
{
    private bool _looking;
    public float fastMovementSpeed = 100f;
    public float fastZoomSensitivity = 50f;
    public float freeLookSensitivity = 5f;
    public float movementSpeed = 50f;
    public float zoomSensitivity = 25f;

    private void Update()
    {
        var fastMode = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
        var speedMovement = fastMode ? fastMovementSpeed : movementSpeed;

        if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow))
            transform.position += -transform.right * speedMovement * Time.deltaTime;

        if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow))
            transform.position += transform.right * speedMovement * Time.deltaTime;

        if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow))
            transform.position += transform.forward * speedMovement * Time.deltaTime;

        if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow))
            transform.position += -transform.forward * speedMovement * Time.deltaTime;

        if (Input.GetKey(KeyCode.Q))
            transform.position += transform.up * speedMovement * Time.deltaTime;

        if (Input.GetKey(KeyCode.E))
            transform.position += -transform.up * speedMovement * Time.deltaTime;

        if (Input.GetKey(KeyCode.R) || Input.GetKey(KeyCode.PageUp))
            transform.position += Vector3.up * speedMovement * Time.deltaTime;

        if (Input.GetKey(KeyCode.F) || Input.GetKey(KeyCode.PageDown))
            transform.position += -Vector3.up * speedMovement * Time.deltaTime;

        if (_looking)
        {
            var newRotationX = transform.localEulerAngles.y + Input.GetAxis("Mouse X") * freeLookSensitivity;
            var newRotationY = transform.localEulerAngles.x - Input.GetAxis("Mouse Y") * freeLookSensitivity;
            transform.localEulerAngles = new Vector3(newRotationY, newRotationX, 0f);
        }

        var axis = Input.GetAxis("Mouse ScrollWheel");
        if (axis != 0)
        {
            var sensitivityZoom = fastMode ? fastZoomSensitivity : zoomSensitivity;
            transform.position += transform.forward * axis * sensitivityZoom;
        }

        if (Input.GetKeyDown(KeyCode.Mouse1))
            StartLooking();
        else if (Input.GetKeyUp(KeyCode.Mouse1)) StopLooking();
    }

    private void OnDisable()
    {
        StopLooking();
    }

    public void StartLooking()
    {
        _looking = true;
        Cursor.visible = false;
        Cursor.lockState = CursorLockMode.Locked;
    }

    public void StopLooking()
    {
        _looking = false;
        Cursor.visible = true;
        Cursor.lockState = CursorLockMode.None;
    }
}