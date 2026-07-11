/** Minimal typings for the vendored js-pytorch browser build. */
declare module "js-pytorch" {
  export const torch: {
    tensor: (data: unknown, requires_grad?: boolean, device?: string) => any;
    ones: (shape: number[], requires_grad?: boolean, device?: string) => any;
    zeros: (shape: number[], requires_grad?: boolean, device?: string) => any;
    // Browser build: randn(shape, requires_grad?, xavier?, device?)
    randn: (
      shape: number[],
      requires_grad?: boolean,
      xavier?: boolean,
      device?: string,
    ) => any;
    matmul: (a: any, b: any) => any;
    optim: {
      Adam: new (params: any[], lr?: number, reg?: number) => {
        step: () => void;
        zero_grad: () => void;
        lr: number;
      };
    };
  };
}
