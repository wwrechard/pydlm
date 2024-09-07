from pydlm.modeler.component import component

import unittest


class testComponent(unittest.TestCase):
    def testCreation(self):
        test_component = component()

        self.assertTrue(hasattr(test_component, "d"))
        self.assertTrue(hasattr(test_component, "name"))
        self.assertTrue(hasattr(test_component, "componentType"))
        self.assertTrue(hasattr(test_component, "discount"))
        self.assertTrue(hasattr(test_component, "evaluation"))
        self.assertTrue(hasattr(test_component, "transition"))
        self.assertTrue(hasattr(test_component, "covPrior"))
        self.assertTrue(hasattr(test_component, "meanPrior"))

    def testEqual(self):
        component_1 = component()
        component_2 = component()

        component_1.d = 1
        component_1.name = "component_1"
        component_1.discount = 1

        component_2.d = 1
        component_2.name = "component_2"
        component_2.discount = 1
        self.assertTrue(component_1 != component_2)

        component_2.name = "component_1"
        self.assertTrue(component_1 == component_2)


if __name__ == "__main__":
    unittest.main()
