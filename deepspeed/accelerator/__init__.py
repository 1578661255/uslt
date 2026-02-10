 
class FakeAccelerator:
    def device_name(self):
        return "cpu"

    def current_device(self):
        return "cpu"

    def is_available(self):
        return False


def get_accelerator():
    return FakeAccelerator()
